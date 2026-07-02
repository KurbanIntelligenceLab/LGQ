"""
MaskGIT Transformer Training with Custom VQ Models.

Trains the MaskGIT bidirectional transformer using our trained quantizers
(VQ, FSQ, LFQ, SimVQ, LMB) as the first-stage tokenizer.

Usage:
    python train_maskgit_custom.py --quantizer vq --epochs 100
    python train_maskgit_custom.py --quantizer sim_vq --epochs 100
"""

import os
import sys
import csv
import random
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import utils as vutils
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image


def _ddp_env():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    return True, rank, local_rank, world_size


def _seed_everything(seed):
    """Seed python/numpy/torch (+cuda) for reproducible multi-seed runs.
    Seeded identically on all DDP ranks: DDP broadcasts rank-0 params at wrap
    time, and the only stochastic data op (mask sampling) is per-shard, so a
    shared seed is correct and avoids rank-divergent init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _NullLogger:
    def add_scalar(self, *a, **kw): pass
    def add_image(self, *a, **kw): pass
    def add_histogram(self, *a, **kw): pass
    def flush(self): pass
    def close(self): pass

# Add parent directory
sys.path.insert(0, '..')

from bidirectional_transformer import BidirectionalTransformer
from vq_adapter import load_vq_adapter


# -----------------------------------------------------------------------------
# Data loading (adapted for 128x128)
# -----------------------------------------------------------------------------
class ImagePaths(Dataset):
    """Dataset for loading images at specified resolution.

    Returns (image_tensor, class_idx). When ``class_map`` is provided (a dict
    mapping parent-dir name -> int label), the dataset operates in
    class-conditional mode and assigns labels from each image's immediate
    parent directory. When ``class_map`` is None and ``num_classes`` is also
    None, labels are all 0 (unconditional fallback for backwards compat).
    """

    def __init__(self, path, size=128, class_map=None):
        self.size = size
        self.images = []
        self.labels = []
        self.class_map = class_map

        # Recursively find all images and label by parent directory name.
        # followlinks=True lets us build ImageNet subset datasets out of
        # symlinked wnid directories (used by smoke tests / sweeps).
        for root, dirs, files in os.walk(path, followlinks=True):
            parent = os.path.basename(root)
            label = -1
            if class_map is not None:
                label = class_map.get(parent, -1)
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    self.images.append(os.path.join(root, file))
                    self.labels.append(label if label >= 0 else 0)

        self._length = len(self.images)
        if class_map is not None:
            n_classed = sum(1 for l in self.labels if l >= 0)
            print(f"Found {self._length} images in {path} "
                  f"(class-labeled: {n_classed}/{self._length})")
        else:
            print(f"Found {self._length} images in {path}")

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        # SmallestMaxSize: resize so smallest side == self.size, preserving aspect
        w, h = image.size
        if w <= h:
            new_w = self.size
            new_h = int(round(h * self.size / w))
        else:
            new_h = self.size
            new_w = int(round(w * self.size / h))
        image = TF.resize(image, [new_h, new_w])
        image = TF.center_crop(image, [self.size, self.size])
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = image.transpose(2, 0, 1)
        return image

    def __getitem__(self, i):
        return self.preprocess_image(self.images[i]), self.labels[i]


def build_imagenet_class_map(path):
    """Build a stable {wnid -> int} mapping by sorting immediate subdirs.

    This matches torchvision.datasets.ImageFolder's class indexing, so labels
    align with standard ImageNet class IDs in the literature.
    """
    if not os.path.isdir(path):
        return None
    # Resolve symlinks here too so symlinked wnid dirs are counted as classes.
    subs = sorted(d for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)))  # isdir follows symlinks
    if not subs:
        return None
    return {name: i for i, name in enumerate(subs)}


# -----------------------------------------------------------------------------
# MaskGIT Transformer Wrapper (adapted for custom VQ models)
# -----------------------------------------------------------------------------
class MaskGITTransformer(torch.nn.Module):
    """MaskGIT-style masked transformer using custom VQ models."""
    
    def __init__(self, args, vq_model):
        super().__init__()
        self.num_image_tokens = args.num_image_tokens
        self.num_codebook_vectors = args.num_codebook_vectors
        self.mask_token_id = args.num_codebook_vectors
        self.sos_token = args.num_codebook_vectors + 1
        # Class tokens occupy ids [num_codebook+2, num_codebook+2+num_classes).
        # num_classes=0 disables conditioning (unconditional fallback).
        self.num_classes = int(getattr(args, "num_classes", 0))
        self.class_offset = args.num_codebook_vectors + 2
        self.choice_temperature = 4.5

        self.gamma = self.gamma_func("cosine")
        
        # Bidirectional transformer
        self.transformer = BidirectionalTransformer(args)
        
        # VQ model (frozen)
        self.vqgan = vq_model
        for param in self.vqgan.parameters():
            param.requires_grad = False
        self.vqgan.eval()
        
        # Get embedding dimension from VQ model for decoding
        self.embedding_dim = vq_model.embedding_dim
        self.spatial_size = int(np.sqrt(self.num_image_tokens))
        
        print(f"Transformer parameters: {sum(p.numel() for p in self.transformer.parameters()):,}")
    
    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def encode_to_z(self, x):
        """Encode images to quantized latents and indices."""
        quant_z, _, (_, _, indices) = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
    def _cond_tokens(self, batch_size, labels, device):
        """Return the (B, 1) tensor of tokens to prepend at position 0.

        Class-cond mode (num_classes>0): label + class_offset, requires labels.
        Unconditional fallback: a constant SOS token.
        """
        if self.num_classes > 0:
            if labels is None:
                raise ValueError("class-conditional MaskGIT requires labels in forward/sample")
            return (labels.to(device=device, dtype=torch.long) + self.class_offset).view(batch_size, 1)
        return torch.full((batch_size, 1), self.sos_token, dtype=torch.long, device=device)

    def forward(self, x, labels=None):
        """Training forward pass with masked token prediction.

        labels: (B,) int tensor of class indices in [0, num_classes). Required
        when num_classes>0; ignored otherwise.
        """
        _, z_indices = self.encode_to_z(x)
        device = z_indices.device
        cond_tokens = self._cond_tokens(x.shape[0], labels, device)

        # Random masking ratio
        r = np.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        r = int(r)

        # Create random mask
        sample = torch.rand(z_indices.shape, device=device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=sample, value=True)

        # Apply mask
        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=device)
        a_indices = mask * z_indices + (~mask) * masked_indices

        # Prepend conditioning token (class id, or SOS in unconditional fallback)
        a_indices = torch.cat((cond_tokens, a_indices), dim=1)
        target = torch.cat((cond_tokens, z_indices), dim=1)

        logits = self.transformer(a_indices)

        return logits, target
    
    @torch.no_grad()
    def forward_val(self, x, labels=None, mask_ratio=0.5, seed=None):
        """
        Validation forward with fixed mask ratio for reproducible NLL and masked accuracy.
        Returns logits, target, and mask (B, num_image_tokens) where True = masked position.
        """
        _, z_indices = self.encode_to_z(x)
        N = z_indices.shape[1]
        device = z_indices.device
        cond_tokens = self._cond_tokens(x.shape[0], labels, device)

        # Fixed number of masked positions
        r = max(1, int(np.floor(mask_ratio * N)))
        if seed is not None:
            g = torch.Generator(device=device).manual_seed(seed)
            perm = torch.rand(z_indices.shape, device=device, generator=g).argsort(dim=1)
        else:
            perm = torch.rand(z_indices.shape, device=device).argsort(dim=1)
        # Mask the first r positions (lowest random values)
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=device)
        mask.scatter_(dim=1, index=perm[:, :r], value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=device)
        a_indices = torch.where(mask, masked_indices, z_indices)
        a_indices = torch.cat((cond_tokens, a_indices), dim=1)
        target = torch.cat((cond_tokens, z_indices), dim=1)

        logits = self.transformer(a_indices)
        return logits, target, mask

    def create_input_tokens_normal(self, num, device="cuda"):
        """Create initial masked tokens for generation."""
        blank_tokens = torch.ones((num, self.num_image_tokens), device=device)
        masked_tokens = self.mask_token_id * blank_tokens
        return masked_tokens.to(torch.int64)
    
    def mask_by_random_topk(self, mask_len, probs, temperature=1.0, device="cuda"):
        """Mask tokens by confidence with Gumbel noise."""
        confidence = torch.log(probs) + temperature * torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        masking = (confidence < cut_off)
        return masking
    
    @staticmethod
    def _filter_logits(logits, top_k=0, top_p=0.0):
        """Apply top-k and/or top-p (nucleus) filtering to a (..., V) logits
        tensor, returning logits with the dropped entries set to -inf so they
        receive zero probability under softmax/Categorical."""
        if top_k and top_k > 0:
            k = min(int(top_k), logits.size(-1))
            kth = torch.topk(logits, k, dim=-1).values[..., -1, None]
            logits = torch.where(logits < kth, torch.full_like(logits, float("-inf")), logits)
        if top_p and 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove = cum > top_p
            # Always keep the top-1 token; shift the mask so the first token
            # that crosses the threshold is retained.
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove = remove.scatter(-1, sorted_idx, remove)
            logits = logits.masked_fill(remove, float("-inf"))
        return logits

    @torch.no_grad()
    def sample(self, num=1, labels=None, T=11, mode="cosine", device="cuda",
               temperature=1.0, temp_anneal=False, temp_min=1.0,
               top_k=0, top_p=0.0):
        """Generate samples using iterative decoding.

        labels: (num,) long tensor of class indices (required when num_classes>0).

        Sampling controls (defaults reproduce the original T=11 pure-sample
        behaviour: temperature=1, no anneal, no truncation):
          temperature : softmax temperature for the token logits. <1 sharpens
                        (more coherent, less diverse), >1 flattens.
          temp_anneal : if True, linearly decay the token temperature from
                        ``temperature`` (first decoding step) to ``temp_min``
                        (last step) -- diverse early, confident late.
          temp_min    : annealing floor (only used when temp_anneal=True).
          top_k       : keep only the top-k logits per position (0 = off).
          top_p       : nucleus filtering, keep the smallest set of tokens whose
                        cumulative prob >= top_p (0 = off).
        These act only on the *value* sampling; the MaskGIT keep/confidence
        ranking still uses the model's raw (untempered) probabilities.
        """
        inputs = self.create_input_tokens_normal(num, device=device)

        cond_tokens = self._cond_tokens(inputs.shape[0], labels, device)
        inputs = torch.cat((cond_tokens, inputs), dim=1)

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs

        _CONFIDENCE_OF_KNOWN_TOKENS = torch.tensor([torch.inf], device=device)

        for t in range(T):
            logits = self.transformer(cur_ids)
            # Restrict sampling to real codebook entries (positions 0..K-1).
            # Mask, SOS, and class tokens must never be sampled into a slot.
            logits = logits[..., :self.num_codebook_vectors]

            ratio = 1. * (t + 1) / T

            # Token-sampling temperature (optionally annealed: hotter early when
            # most slots are masked, cooler late when committing).
            cur_temp = temperature
            if temp_anneal:
                cur_temp = temperature * (1.0 - ratio) + temp_min * ratio
            cur_temp = max(float(cur_temp), 1e-4)
            samp_logits = self._filter_logits(logits / cur_temp, top_k=top_k, top_p=top_p)
            sampled_ids = torch.distributions.categorical.Categorical(logits=samp_logits).sample()

            unknown_map = (cur_ids == self.mask_token_id)
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)

            mask_ratio = gamma(ratio)

            probs = F.softmax(logits, dim=-1)
            safe_ids = sampled_ids.clamp(max=self.num_codebook_vectors - 1)
            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(safe_ids, -1), -1), -1)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)

            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)
            mask_len = torch.maximum(torch.zeros_like(mask_len),
                                     torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True) - 1, mask_len))

            masking = self.mask_by_random_topk(mask_len, selected_probs,
                                               temperature=self.choice_temperature * (1. - ratio),
                                               device=device)
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)

        # Strip conditioning token; clamp to valid codebook range in case any
        # mask tokens survived the final iteration.
        return cur_ids[:, 1:].clamp(max=self.num_codebook_vectors - 1)

    def indices_to_image(self, indices):
        """Convert token indices back to images."""
        H = W = self.spatial_size
        vectors = self.vqgan.codebook.embedding(indices)  # (B, num_tokens, embedding_dim)
        vectors = vectors.view(indices.shape[0], H, W, self.embedding_dim)
        vectors = vectors.permute(0, 3, 1, 2)  # (B, embedding_dim, H, W)
        image = self.vqgan.decode(vectors)
        return image
    
    @torch.no_grad()
    def log_images(self, x, labels=None, mode="cosine", device="cuda"):
        """Generate log images for visualization."""
        log = dict()

        _, z_indices = self.encode_to_z(x)

        # Create new sample (use the real labels so the sample matches the input class)
        index_sample = self.sample(num=x.shape[0], labels=labels, mode=mode, device=device)
        x_new = self.indices_to_image(index_sample)

        # Create half sample (inpainting placeholder — same path for now)
        half_index_sample = self.sample(num=x.shape[0], labels=labels, mode=mode, device=device)
        x_sample = self.indices_to_image(half_index_sample)

        # Create reconstruction
        x_rec = self.indices_to_image(z_indices)

        log["input"] = x
        log["rec"] = x_rec
        log["half_sample"] = x_sample
        log["new_sample"] = x_new
        return log, torch.cat((x, x_rec, x_sample, x_new))


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
class TrainMaskGIT:
    def __init__(self, args):
        self.args = args

        # Detect DDP from torchrun env vars
        self.is_distributed, self.rank, self.local_rank, self.world_size = _ddp_env()
        self.is_main = (self.rank == 0)
        if self.is_distributed:
            torch.cuda.set_device(self.local_rank)
            args.device = f"cuda:{self.local_rank}"
            if not dist.is_initialized():
                dist.init_process_group(backend="nccl")
        self.device = args.device

        # Seed before any model/data construction so multi-seed runs diverge.
        if getattr(args, "seed", None) is not None:
            _seed_everything(args.seed)
            if self.is_main:
                print(f"Seeded everything with seed={args.seed}")

        # Load VQ model
        if self.is_main:
            print(f"\nLoading {args.quantizer} model...")
        self.vq_model = load_vq_adapter(
            args.quantizer,
            args.vq_checkpoint,
            device=args.device,
            image_size=args.image_size,
        )

        # Update num_image_tokens based on VQ model
        args.num_image_tokens = self.vq_model.num_image_tokens
        if self.is_main:
            print(f"Using {args.num_image_tokens} image tokens")

        # Create transformer
        self.model = MaskGITTransformer(args, self.vq_model).to(self.device)
        # Keep an unwrapped handle for direct access (sampling, log_images, save).
        self._raw_model = self.model
        if self.is_distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                find_unused_parameters=False,
            )

        # Optimizer (only train transformer). Parameter identity is preserved
        # by DDP wrapping, so optimizing _raw_model.transformer.parameters()
        # is equivalent to optimizing the wrapped module's transformer params.
        self.optim = torch.optim.AdamW(
            self._raw_model.transformer.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.96),
            weight_decay=4.5e-2
        )
        
        # LR scheduler with warmup
        self.warmup_steps = 1000
        self.scheduler = None  # Will use manual warmup
        
        # Resume state
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.global_step = 0

        # Convergence / early-stop state (tracked on val NLL).
        self.best_val_nll = float('inf')
        self.epochs_since_improve = 0
        
        # Logging (rank 0 owns disk IO)
        self.output_dir = args.output_dir
        if self.is_main:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, "samples"), exist_ok=True)
        if self.is_distributed:
            dist.barrier()

        self.logger = _NullLogger()

        # Resume from checkpoint if specified
        if getattr(args, 'resume', None):
            self._load_resume(args.resume)

        # Start training
        self.train()

        if self.is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    
    def get_lr(self, step):
        """Linear warmup then constant."""
        if step < self.warmup_steps:
            return self.args.learning_rate * step / self.warmup_steps
        return self.args.learning_rate
    
    def _load_resume(self, resume_spec):
        """Load checkpoint for resume. resume_spec: path to .pt, or 'latest', or 'best'."""
        ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        if resume_spec == "latest":
            # Find highest transformer_N.pt (exclude transformer_best.pt)
            candidates = []
            for f in os.listdir(ckpt_dir):
                if f.startswith("transformer_") and f.endswith(".pt") and f != "transformer_best.pt":
                    try:
                        n = int(f.replace("transformer_", "").replace(".pt", ""))
                        candidates.append((n, os.path.join(ckpt_dir, f)))
                    except ValueError:
                        pass
            if not candidates:
                print(f"No latest checkpoint found in {ckpt_dir}, starting from epoch 1")
                return
            path = max(candidates, key=lambda x: x[0])[1]
        elif resume_spec == "best":
            path = os.path.join(ckpt_dir, "transformer_best.pt")
            if not os.path.isfile(path):
                print(f"No best checkpoint at {path}, starting from epoch 1")
                return
        else:
            path = resume_spec
            if not os.path.isfile(path):
                print(f"Resume path not found: {path}, starting from epoch 1")
                return
        if self.is_main:
            print(f"Resuming from checkpoint: {path}")
        try:
            ckpt = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=self.device)
        self._raw_model.transformer.load_state_dict(ckpt["transformer_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            self.optim.load_state_dict(ckpt["optimizer_state_dict"])
        epoch = ckpt.get("epoch", 0)
        if isinstance(epoch, str):
            epoch = 0
        self.start_epoch = epoch + 1
        self.best_loss = ckpt.get("loss", float('inf'))
        if "global_step" in ckpt:
            self.global_step = ckpt["global_step"]
        if self.is_main:
            print(f"Resumed: next epoch {self.start_epoch}, best_loss {self.best_loss:.4f}")
    
    def evaluate_val(self, val_loader, mask_ratio=0.5, seed=0):
        """Compute validation NLL (cross-entropy) and masked token accuracy.
        Under DDP each rank consumes its shard via DistributedSampler;
        accumulators are all-reduced to global totals before averaging.
        """
        self.model.eval()
        nll_sum = 0.0
        nll_count = 0
        masked_correct = 0
        masked_total = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                logits, target, mask = self._raw_model.forward_val(
                    imgs, labels=labels, mask_ratio=mask_ratio, seed=seed
                )
                # Mirror training: exclude the conditioning position from NLL.
                target_for_nll = target.clone()
                target_for_nll[:, 0] = -100
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_for_nll.reshape(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                nll_sum += nll.item()
                nll_count += (target_for_nll != -100).sum().item()
                pred = logits.argmax(dim=-1)
                pred_tokens = pred[:, 1:]
                target_tokens = target[:, 1:]
                correct = (pred_tokens == target_tokens) & mask
                masked_correct += correct.sum().item()
                masked_total += mask.sum().item()
        self.model.train()
        if self.is_distributed:
            t = torch.tensor(
                [nll_sum, float(nll_count), float(masked_correct), float(masked_total)],
                device=self.device, dtype=torch.float64,
            )
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            nll_sum, nll_count, masked_correct, masked_total = t.tolist()
        nll_mean = nll_sum / max(1, nll_count)
        masked_acc = masked_correct / max(1, masked_total)
        return nll_mean, masked_acc

    def train(self):
        args = self.args

        # Build a stable wnid -> class index map from the training dirs so
        # train and val datasets share the same label space.
        class_map = None
        if int(getattr(args, "num_classes", 0)) > 0:
            class_map = build_imagenet_class_map(args.dataset_path)
            if class_map is None:
                raise RuntimeError(
                    f"--num-classes={args.num_classes} but no class subdirs found under {args.dataset_path}"
                )
            if len(class_map) != args.num_classes and self.is_main:
                print(f"  WARNING: found {len(class_map)} class dirs but --num-classes={args.num_classes}")

        # Load data
        full_data = ImagePaths(args.dataset_path, size=args.image_size, class_map=class_map)
        val_loader = None
        val_data = None
        if args.val_dataset_path and os.path.isdir(args.val_dataset_path):
            val_data = ImagePaths(args.val_dataset_path, size=args.image_size, class_map=class_map)
            train_data = full_data
            if self.is_main and len(val_data) > 0:
                print(f"Validation: {len(val_data)} images from {args.val_dataset_path}")
            if len(val_data) == 0:
                val_data = None
        elif getattr(args, "val_ratio", 0.1) > 0 and len(full_data) > 0:
            n = len(full_data)
            n_val = max(1, int(n * args.val_ratio))
            n_train = n - n_val
            train_indices = list(range(0, n_train))
            val_indices = list(range(n_train, n))
            train_data = Subset(full_data, train_indices)
            val_data = Subset(full_data, val_indices)
            if self.is_main:
                print(f"Validation: {n_val} images ({args.val_ratio*100:.0f}% of dataset)")
        else:
            train_data = full_data

        if self.is_distributed:
            train_sampler = DistributedSampler(
                train_data, num_replicas=self.world_size, rank=self.rank,
                shuffle=True, drop_last=True,
            )
        else:
            train_sampler = None
        train_loader = DataLoader(
            train_data,
            batch_size=args.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        if val_data is not None:
            if self.is_distributed:
                val_sampler = DistributedSampler(
                    val_data, num_replicas=self.world_size, rank=self.rank,
                    shuffle=False, drop_last=False,
                )
            else:
                val_sampler = None
            val_loader = DataLoader(
                val_data,
                batch_size=args.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
        
        global_step = self.global_step
        best_loss = self.best_loss
        lr = self.args.learning_rate  # Initialize lr
        val_mask_ratio = getattr(args, "val_mask_ratio", 0.5)

        for epoch in range(self.start_epoch, args.epochs + 1):
            if self.is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.model.train()
            epoch_losses = []

            iterator = train_loader
            if self.is_main:
                iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
            for i, batch in enumerate(iterator):
                imgs, labels = batch
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits, target = self.model(imgs, labels=labels)
                # Ignore the conditioning position at index 0 in the loss --
                # class id (or SOS) is the input, not something to predict.
                target_for_loss = target.clone()
                target_for_loss[:, 0] = -100
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_for_loss.reshape(-1),
                    ignore_index=-100,
                )

                loss.backward()

                if (i + 1) % args.accum_grad == 0:
                    lr = self.get_lr(global_step)
                    for param_group in self.optim.param_groups:
                        param_group['lr'] = lr

                    torch.nn.utils.clip_grad_norm_(self._raw_model.transformer.parameters(), 1.0)

                    self.optim.step()
                    self.optim.zero_grad()
                    global_step += 1
                    self.global_step = global_step

                epoch_losses.append(loss.item())
                if self.is_main:
                    iterator.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr:.2e}")

                if global_step % 100 == 0:
                    self.logger.add_scalar("train/loss", loss.item(), global_step)
                    self.logger.add_scalar("train/lr", lr, global_step)

                # Smoke-test knob: cap optimizer steps to exercise the pipeline fast.
                if getattr(args, "max_steps", 0) and global_step >= args.max_steps:
                    break

            # Average loss across all ranks for fair best-tracking under DDP.
            local_avg = float(np.mean(epoch_losses)) if epoch_losses else 0.0
            if self.is_distributed:
                t = torch.tensor([local_avg], device=self.device, dtype=torch.float64)
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                avg_loss = (t.item() / self.world_size)
            else:
                avg_loss = local_avg
            if self.is_main:
                print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")
            self.logger.add_scalar("train/epoch_loss", avg_loss, epoch)

            val_nll, val_masked_acc = (float("nan"), float("nan"))
            stop = False
            if val_loader is not None:
                val_nll, val_masked_acc = self.evaluate_val(
                    val_loader, mask_ratio=val_mask_ratio, seed=epoch
                )
                self.logger.add_scalar("val/nll", val_nll, epoch)
                self.logger.add_scalar("val/masked_accuracy", val_masked_acc, epoch)
                if self.is_main:
                    print(f"  Val NLL: {val_nll:.4f}  Val masked acc: {val_masked_acc:.4f}")

                # Track val-NLL plateau for convergence / optional early stop.
                # val_nll is already all-reduced, so every rank agrees.
                if val_nll < self.best_val_nll - args.early_stop_min_delta:
                    self.best_val_nll = val_nll
                    self.epochs_since_improve = 0
                    if self.is_main:
                        self.save_checkpoint("bestval", avg_loss)
                else:
                    self.epochs_since_improve += 1
                if args.early_stop_patience > 0 and \
                        self.epochs_since_improve >= args.early_stop_patience:
                    stop = True

            # Persist per-epoch history so the plateau is demonstrable on disk.
            if self.is_main:
                self._append_history({
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": f"{avg_loss:.6f}",
                    "val_nll": "" if val_loader is None else f"{val_nll:.6f}",
                    "val_masked_acc": "" if val_loader is None else f"{val_masked_acc:.6f}",
                    "lr": f"{lr:.3e}",
                    "best_val_nll": "" if val_loader is None else f"{self.best_val_nll:.6f}",
                })

            if epoch % args.sample_every == 0 and self.is_main:
                self._raw_model.eval()
                try:
                    sample_imgs = imgs[:4]
                    sample_labels = labels[:4] if self._raw_model.num_classes > 0 else None
                    log, vis = self._raw_model.log_images(
                        sample_imgs, labels=sample_labels, device=self.device
                    )
                    vis = (vis + 1) / 2
                    vutils.save_image(
                        vis,
                        os.path.join(self.output_dir, "samples", f"epoch_{epoch:04d}.png"),
                        nrow=4
                    )
                except Exception as e:
                    print(f"Sample generation failed: {e}")

            if epoch % args.save_every == 0 and self.is_main:
                self.save_checkpoint(epoch, avg_loss)

            if avg_loss < best_loss:
                best_loss = avg_loss
                if self.is_main:
                    self.save_checkpoint("best", avg_loss)

            if self.is_distributed:
                dist.barrier()

            if stop:
                if self.is_main:
                    print(f"\nEarly stop at epoch {epoch}: val NLL plateaued "
                          f"(no improvement for {self.epochs_since_improve} epochs, "
                          f"best={self.best_val_nll:.4f}).")
                    self.save_checkpoint(epoch, avg_loss)
                break

        if self.is_main:
            print(f"\nTraining complete. Best loss: {best_loss:.4f}")

    def _append_history(self, row):
        """Append one epoch's metrics to {output_dir}/train_history.csv (rank 0).
        Demonstrates the val-NLL plateau; written every epoch, resume-safe."""
        path = os.path.join(self.output_dir, "train_history.csv")
        cols = ["epoch", "global_step", "train_loss", "val_nll",
                "val_masked_acc", "lr", "best_val_nll"]
        new = not os.path.isfile(path)
        with open(path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerow(row)

    def save_checkpoint(self, epoch, loss):
        ckpt = {
            "epoch": epoch,
            "global_step": getattr(self, "global_step", 0),
            "transformer_state_dict": self._raw_model.transformer.state_dict(),
            "optimizer_state_dict": self.optim.state_dict(),
            "loss": loss,
            "args": vars(self.args),
        }
        path = os.path.join(self.output_dir, "checkpoints", f"transformer_{epoch}.pt")
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def get_default_vq_checkpoint(quantizer):
    """Get default checkpoint path for a quantizer type."""
    # Use absolute path based on script location
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(script_dir, "..", "results")
    paths = {
        "fsq": os.path.join(base, "fsq/fsq_cb16k/checkpoints/latest_model.pt"),
        "lfq": os.path.join(base, "lfq/lfq_cb16k/checkpoints/best_model.pt"),
        "sim_vq": os.path.join(base, "sim_vq/sim_vq_cb16k/checkpoints/best_model.pt"),
        "lmb_fixed": os.path.join(base, "lmb/lmb_fixed_init/checkpoints/latest_model.pt"),  # LMB-Fixed (main)
        "lmb_perchannel": os.path.join(base, "lmb/lmb_ablation_perchannel_fair/checkpoints/best_model.pt"),
    }
    # VQ: try vq_cb16k first, then any results/vq/<subdir>/checkpoints/*.pt
    if quantizer == "vq":
        vq_candidates = [
            os.path.join(base, "vq/vq_cb16k/checkpoints/best_model.pt"),
            os.path.join(base, "vq/vq_cb16k/checkpoints/latest_model.pt"),
        ]
        vq_dir = os.path.join(base, "vq")
        if os.path.isdir(vq_dir):
            for sub in os.listdir(vq_dir):
                for name in ("best_model.pt", "latest_model.pt"):
                    ckpt = os.path.join(vq_dir, sub, "checkpoints", name)
                    if os.path.isfile(ckpt) and ckpt not in vq_candidates:
                        vq_candidates.append(ckpt)
        for p in vq_candidates:
            if os.path.isfile(p):
                return p
        return vq_candidates[0]
    # LMB: try several possible run dirs (flattened_main, fixed_init, or any run with checkpoints)
    if quantizer == "lmb":
        lmb_candidates = [
            os.path.join(base, "lmb/lmb_flattened_main/checkpoints/latest_model.pt"),
            os.path.join(base, "lmb/lmb_fixed_init/checkpoints/latest_model.pt"),
        ]
        lmb_dir = os.path.join(base, "lmb")
        if os.path.isdir(lmb_dir):
            for sub in os.listdir(lmb_dir):
                ckpt = os.path.join(lmb_dir, sub, "checkpoints", "latest_model.pt")
                if os.path.isfile(ckpt) and ckpt not in lmb_candidates:
                    lmb_candidates.append(ckpt)
        for p in lmb_candidates:
            if os.path.isfile(p):
                return p
        return lmb_candidates[0]  # default to first so --vq-checkpoint is still required if missing
    return paths.get(quantizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MaskGIT with custom VQ")
    
    # VQ model
    parser.add_argument("--quantizer", type=str, default="vq",
                        choices=["vq", "fsq", "lfq", "sim_vq", "softvq", "lmb", "lmb_fixed", "ibq", "rot_vq"],
                        help="Quantizer type to use")
    parser.add_argument("--vq-checkpoint", type=str, default=None,
                        help="Path to VQ model checkpoint")
    
    # Data - use absolute path
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    _default_data = _os.path.join(_script_dir, "..", "data/imagenet")
    parser.add_argument("--dataset-path", type=str, default=_default_data,
                        help="Path to training images")
    parser.add_argument("--val-dataset-path", type=str, default=None,
                        help="Path to validation images (if None, use --val-ratio split from train)")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of data for validation when --val-dataset-path not set (0 = no val)")
    parser.add_argument("--val-mask-ratio", type=float, default=0.5,
                        help="Fixed mask ratio for validation NLL / masked accuracy")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Image size (must match VQ model)")
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Training
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accum-grad", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    
    # Transformer architecture
    parser.add_argument("--num-codebook-vectors", type=int, default=16384,
                        help="VQ codebook size")
    parser.add_argument("--num-classes", type=int, default=0,
                        help="Number of class-conditioning tokens (1000 for ImageNet). "
                             "0 disables conditioning (unconditional fallback).")
    parser.add_argument("--dim", type=int, default=512,
                        help="Transformer dimension")
    parser.add_argument("--hidden-dim", type=int, default=2048,
                        help="Transformer MLP hidden dimension")
    parser.add_argument("--n-layers", type=int, default=12,
                        help="Number of transformer layers")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--sample-every", type=int, default=5,
                        help="Generate samples every N epochs")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint: path to .pt, or 'latest', or 'best'")

    # Reproducibility / convergence
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (python/numpy/torch/cuda). None = nondeterministic.")
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop if val NLL doesn't improve for N epochs (0 = disabled; "
                             "train to --epochs and show plateau post-hoc).")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                        help="Minimum val-NLL improvement to reset early-stop patience.")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Cap optimizer steps (0 = unlimited). For smoke tests only.")

    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()

    # cuDNN disabled: module cuda toolkit (11.x) lacks H100 sm_90 kernels for
    # every bf16/fp32 conv shape, segfaulting on the encoder stem. Native ATen
    # conv works. Re-enable only after upgrading to a CUDA 12.x PyTorch build.
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = False

    # Set default VQ checkpoint
    if args.vq_checkpoint is None:
        args.vq_checkpoint = get_default_vq_checkpoint(args.quantizer)
    
    # Set default output directory (use absolute path)
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.output_dir = os.path.join(script_dir, "..", "results", "maskgit", args.quantizer)
    
    # Placeholder for num_image_tokens (will be set from VQ model)
    args.num_image_tokens = 64  # Default, will be updated
    
    _is_main_proc = int(os.environ.get("RANK", "0")) == 0
    if _is_main_proc:
        print("=" * 60)
        print("MaskGIT Training Configuration")
        print("=" * 60)
        print(f"Quantizer: {args.quantizer}")
        print(f"VQ Checkpoint: {args.vq_checkpoint}")
        print(f"Dataset: {args.dataset_path}")
        print(f"Image Size: {args.image_size}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Transformer: {args.n_layers} layers, dim={args.dim}")
        print(f"Output: {args.output_dir}")
        print("=" * 60)

    trainer = TrainMaskGIT(args)

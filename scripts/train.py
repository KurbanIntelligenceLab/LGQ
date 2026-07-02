#!/usr/bin/env python3
"""
Unified training script for all quantization models on ImageNet.

All models use the same:
- Data loading and transforms
- Training loop
- Evaluation after each epoch
- Checkpointing
- Logging
- Encoder/Decoder backbone

Only the quantization layer differs between models.

Supports multiple concurrent runs with unique directories based on hyperparameters.

Usage:
    python scripts/train.py --model fsq --levels 8 5 5 5
    python scripts/train.py --model vq --codebook-size 8192
    python scripts/train.py --model lfq --codebook-size 65536
    python scripts/train.py --model sim_vq --codebook-size 8192 --use-mlp
    python scripts/train.py --model lgq --num-bins 128
    
    # Custom run name
    python scripts/train.py --model fsq --run-name my_experiment
"""

import argparse
import json
import hashlib
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from PIL import Image


def is_dist_env() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def get_rank() -> int:
    return int(os.environ.get("RANK", 0))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def is_main() -> bool:
    return get_rank() == 0


def unwrap(m):
    return m.module if isinstance(m, DDP) else m


def _quantizer_needs_lazy_init(model) -> bool:
    """True if the model's quantizer populates `self.centers` lazily on the
    first forward pass (via `_maybe_init_from_batch`). DDP breaks this because
    each rank would init against a different shard/seed, so we trigger it on
    rank 0 before wrapping."""
    q = getattr(unwrap(model), "quantize", None)
    return q is not None and hasattr(q, "_inited") and int(q._inited.item()) == 0


def rprint(*a, **kw):
    if is_main():
        print(*a, **kw)

from scripts.evaluation.metrics_logging import CsvMetricLogger
from scripts.evaluation.fid import FIDCalculator
from scripts.evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric, compute_codebook_utilization
from configs import MODEL_CONFIGS
from configs.base import LossOutput
from quantization.model import ModelConfig, count_model_parameters
from losses.discriminator import PatchGANDiscriminator, hinge_loss_d, hinge_loss_g, calculate_adaptive_weight
from losses.perceptual import PerceptualLoss

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips corrupt/unreadable images instead of crashing."""

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                return super().__getitem__(idx)
            except Exception as e:
                path, _ = self.samples[idx]
                print(f"WARNING: skipping corrupt image {path}: {e}")
                idx = (idx + 1) % len(self)
        raise RuntimeError("Too many consecutive corrupt images")


class FlatImageDataset(Dataset):
    """
    Simple image dataset that loads all images from a directory.

    Unlike ImageFolder, this doesn't require class subdirectories.
    Perfect for reconstruction tasks where we don't need labels.
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform

        # Find all image files
        self.image_paths = []
        root_path = Path(root)

        for ext in self.EXTENSIONS:
            self.image_paths.extend(root_path.glob(f'*{ext}'))
            self.image_paths.extend(root_path.glob(f'*{ext.upper()}'))

        # Also search subdirectories (for ImageNet-style class folders)
        for ext in self.EXTENSIONS:
            self.image_paths.extend(root_path.glob(f'**/*{ext}'))
            self.image_paths.extend(root_path.glob(f'**/*{ext.upper()}'))

        # Remove duplicates and sort for reproducibility
        self.image_paths = sorted(set(self.image_paths))

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        # Return dummy label (0) for compatibility with training loop
        return img, 0


# ImageNet-100: first 100 synsets in sorted order
IMAGENET100_SYNSETS = None  # lazily populated

def get_imagenet100_synsets(data_root: str):
    """Return the first 100 synset folder names (sorted) from the training set."""
    global IMAGENET100_SYNSETS
    if IMAGENET100_SYNSETS is not None:
        return IMAGENET100_SYNSETS
    train_dir = os.path.join(data_root, "train")
    all_synsets = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    IMAGENET100_SYNSETS = set(all_synsets[:100])
    return IMAGENET100_SYNSETS


class ImageNet100Folder(SafeImageFolder):
    """ImageFolder filtered to the first 100 synsets (ImageNet-100)."""

    def __init__(self, root, transform=None, synsets=None):
        super().__init__(root, transform=transform)
        if synsets is not None:
            # Filter samples to only include the specified synsets
            self.samples = [(p, t) for p, t in self.samples
                            if os.path.basename(os.path.dirname(p)) in synsets]
            self.targets = [t for _, t in self.samples]
            self.imgs = self.samples


def generate_run_name(args: argparse.Namespace) -> str:
    """
    Generate a unique run name based on hyperparameters.
    
    Format: {model}_{key_params}_{timestamp}_{hash}
    
    Examples:
        fsq_lv8-5-5-5_bs64_lr3e-04_20260115_143052_a1b2
        vq_cb8192_bs64_lr3e-04_20260115_143052_c3d4
        lgq_nb128_bs64_lr3e-04_20260115_143052_e5f6
    """
    parts = [args.model]
    
    # Model-specific key parameters
    if args.model in ("fsq", "ifsq"):
        levels_str = "-".join(str(l) for l in args.levels)
        parts.append(f"lv{levels_str}")
    elif args.model == "vq":
        parts.append(f"cb{args.codebook_size}")
    elif args.model == "lfq":
        parts.append(f"cb{args.codebook_size}")
        if args.spherical:
            parts.append("sph")
    elif args.model == "sim_vq":
        parts.append(f"cb{args.codebook_size}")
        if args.use_mlp:
            parts.append("mlp")
    elif args.model == "rot_vq":
        parts.append(f"cb{args.codebook_size}")
    elif args.model in ("lgq", "lmb"):
        parts.append(f"nb{args.num_bins}")
        if hasattr(args, 'tau_start') and hasattr(args, 'tau_end'):
            if args.tau_start != args.tau_end:
                parts.append(f"tau{args.tau_start}-{args.tau_end}")
    
    # Common training parameters
    parts.append(f"bs{args.batch_size}")
    parts.append(f"lr{args.lr:.0e}".replace("+", "").replace("0", ""))
    parts.append(f"dim{args.dim}")
    
    # Timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
    # Short hash of full config for uniqueness
    config_str = json.dumps(vars(args), sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:4]
    parts.append(config_hash)
    
    return "_".join(parts)


def create_common_parser() -> argparse.ArgumentParser:
    """Create parser with common arguments for all models."""
    parser = argparse.ArgumentParser(
        description="Train quantization model on ImageNet (unified training script)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Model selection
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model type to train")
    
    # Data arguments
    parser.add_argument("--data-root", type=str, default="~/data/imagenet",
                        help="Path to ImageNet dataset")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "imagenet100", "cifar10"],
                        help="Dataset to use: imagenet, imagenet100, or cifar10")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--max-train-samples", type=int, default=0,
                        help="If >0, train on a fixed random subset of this many images "
                             "(seeded by --seed). Used for small-sample ablations.")

    # Training arguments
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use (ignored under DDP — LOCAL_RANK is used instead)")
    parser.add_argument("--warmup-steps", type=int, default=0,
                        help="Linear warmup steps for LR schedule (0 = disabled)")
    parser.add_argument("--lr-schedule", type=str, default="none",
                        choices=["none", "cosine"],
                        help="LR schedule applied after warmup")
    parser.add_argument("--min-lr", type=float, default=0.0,
                        help="Minimum LR floor for cosine schedule")
    
    # Architecture arguments (shared by all models)
    parser.add_argument("--dim", type=int, default=256,
                        help="Feature dimension (num_hiddens)")
    parser.add_argument("--embedding-dim", type=int, default=64,
                        help="Embedding dimension / latent channels (default: 64, matches paper)")
    parser.add_argument("--compression-factor", type=int, default=16,
                        help="Spatial compression factor (8 or 16)")
    parser.add_argument("--num-residual-layers", type=int, default=2,
                        help="Number of residual layers in encoder/decoder")
    parser.add_argument("--legacy-decoder", action="store_true",
                        help="Use old decoder (ch = max(32, embedding_dim)) for reproducing pre-decfix runs")

    # Output arguments
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: auto-generated from hyperparams)")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Custom run name (default: auto-generated)")
    parser.add_argument("--results-root", type=str, default="results",
                        help="Root directory for all results")
    
    # Evaluation arguments
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Batch size for evaluation (default: same as training)")
    parser.add_argument("--eval-samples", type=int, default=None,
                        help="Number of samples for evaluation (default: full val set)")
    parser.add_argument("--no-eval", action="store_true",
                        help="Skip evaluation after each epoch")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Input image size (default: 128, matches paper)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint (path to checkpoint file, or 'latest' or 'best')")
    parser.add_argument("--keep-only-best-latest", action="store_true",
                        help="Save only best_model.pt and latest_model.pt (no per-epoch checkpoint_epoch_*.pt) to save space")
    parser.add_argument("--save-initial-centers", action="store_true",
                        help="Save LGQ bin centers after epoch 1 to analysis_epoch001_centers.pt for first→last comparison plots")
    
    # FID arguments
    parser.add_argument("--fid-samples", type=int, default=10000,
                        help="Number of samples for rFID computation")
    parser.add_argument("--no-rfid", action="store_true",
                        help="Skip rFID computation during epoch eval (saves ~50 min/epoch on the image-saving phase)")
    
    # Logging arguments
    parser.add_argument("--csv-log-every", type=int, default=50,
                        help="Write a row to train_metrics.csv every N iterations (0 to disable)")
    parser.add_argument("--print-every", type=int, default=100,
                        help="Print training progress every N iterations (0 to disable)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="Stop each epoch after this many steps (0 = full epoch; for smoke tests)")
    parser.add_argument("--grad-accum-steps", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch_size * grad_accum_steps)")
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision (bf16 on A100+, fp16 otherwise)")
    parser.add_argument("--codebook-init", type=str, default=None,
                        help="Path to .npy file with pre-computed codebook (from phase0_datacentric.py)")

    # Perceptual + adversarial loss arguments
    parser.add_argument("--perceptual-weight", type=float, default=0.0,
                        help="LPIPS perceptual loss weight (0 = disabled)")
    parser.add_argument("--gan-weight", type=float, default=0.0,
                        help="Adversarial loss weight (0 = disabled)")
    parser.add_argument("--disc-start", type=int, default=10000,
                        help="Global step to start discriminator training")
    parser.add_argument("--disc-warmup-steps", type=int, default=0,
                        help="Linear ramp of gan_weight from 0 to full over this many steps after disc_start (0 = no ramp)")
    parser.add_argument("--disc-lr", type=float, default=3e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--disc-n-layers", type=int, default=3,
                        help="Number of PatchGAN discriminator layers")
    parser.add_argument("--l1-weight", type=float, default=1.0,
                        help="L1 reconstruction loss weight (allows rebalancing with perceptual/GAN)")

    # Wandb arguments
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="vector-quantize",
                        help="W&B project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
                        help="W&B entity (team or username)")

    return parser


def _denormalize(x: torch.Tensor, device) -> torch.Tensor:
    """Undo ImageNet normalization to get [0, 1] images."""
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (x * std + mean).clamp(0.0, 1.0)


def train_epoch(
    model,
    model_config,
    train_loader,
    optimizer,
    device,
    *,
    epoch: int,
    global_step: int,
    csv_logger: Optional[CsvMetricLogger],
    csv_log_every: int,
    print_every: int,
    max_steps: int = 0,
    run_name: str = "",
    grad_accum_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    amp_dtype: Optional[torch.dtype] = None,
    # Perceptual + adversarial loss components (None = disabled)
    perceptual_loss_fn: Optional[PerceptualLoss] = None,
    discriminator: Optional[PatchGANDiscriminator] = None,
    opt_disc: Optional[torch.optim.Optimizer] = None,
    perceptual_weight: float = 0.0,
    gan_weight: float = 0.0,
    l1_weight: float = 1.0,
    disc_start: int = 10000,
    disc_warmup_steps: int = 0,
    lr_scheduler=None,
    wandb_run=None,
    lfq_entropy_warmup_steps: int = 0,
    lfq_entropy_warmup_factor: float = 1.0,
):
    """Train for one epoch using model-specific loss computation."""
    model.train()
    if discriminator is not None:
        discriminator.train()
    use_gan = discriminator is not None and gan_weight > 0
    use_percep = perceptual_loss_fn is not None and perceptual_weight > 0

    total_rec_loss = 0.0
    total_aux_loss = 0.0
    total_percep_loss = 0.0
    total_g_loss = 0.0
    total_d_loss = 0.0
    total_active_codes = 0
    num_batches = 0
    aux_loss_name = "aux_loss"
    use_amp = amp_dtype is not None

    prefix = f"Training [{run_name}]" if run_name else "Training"
    num_iters = min(len(train_loader), max_steps) if max_steps > 0 else len(train_loader)
    optimizer.zero_grad()
    if opt_disc is not None:
        opt_disc.zero_grad()

    for batch_idx, (x, _) in enumerate(train_loader):
        if max_steps > 0 and batch_idx >= max_steps:
            break
        x = x.to(device)
        is_accum_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == num_iters
        disc_active = use_gan and global_step >= disc_start

        # LFQ entropy-weight annealing (MAGVIT-v2 schedule):
        # weight = base * (1 + (factor - 1) * max(0, 1 - step/warmup_steps))
        # i.e. linearly decay from base*factor at step 0 to base at warmup_steps,
        # then hold at base. No-op when warmup_steps == 0 or factor == 1.
        if lfq_entropy_warmup_steps > 0 and lfq_entropy_warmup_factor != 1.0:
            inner = unwrap(model) if 'unwrap' in globals() else (model.module if hasattr(model, 'module') else model)
            quant = getattr(inner, 'quantize', None)
            if quant is not None and hasattr(quant, 'entropy_loss_weight'):
                base_w = getattr(inner, '_lfq_entropy_weight_base', None)
                if base_w is None:
                    base_w = float(quant.entropy_loss_weight)
                    inner._lfq_entropy_weight_base = base_w
                progress = min(1.0, max(0.0, global_step / lfq_entropy_warmup_steps))
                cur_factor = lfq_entropy_warmup_factor + (1.0 - lfq_entropy_warmup_factor) * progress
                quant.entropy_loss_weight = base_w * cur_factor

        # ---- Generator step ----
        with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
            loss_output: LossOutput = model_config.compute_loss(model, x)

            # Start with model-specific loss (L1 rec + aux)
            gen_total = l1_weight * loss_output.rec_loss + loss_output.aux_loss

            p_loss_val = 0.0
            g_loss_val = 0.0

            if use_percep or disc_active:
                # Denormalize for perceptual/GAN (they expect [0,1])
                x_01 = _denormalize(x, device)
                recon_01 = _denormalize(loss_output.recon, device)

            if use_percep:
                p_loss = perceptual_loss_fn(x_01, recon_01)
                gen_total = gen_total + perceptual_weight * p_loss
                p_loss_val = float(p_loss.item())

            if disc_active:
                logits_fake = discriminator(recon_01)
                g_loss = hinge_loss_g(logits_fake)
                # Adaptive weight: balance rec grad vs gan grad at last decoder layer
                try:
                    adapt_w = calculate_adaptive_weight(
                        loss_output.rec_loss, g_loss, unwrap(model).dec.out.weight
                    )
                except RuntimeError:
                    adapt_w = torch.tensor(1.0, device=device)
                if disc_warmup_steps > 0:
                    ramp = min(1.0, max(0, global_step - disc_start) / disc_warmup_steps)
                else:
                    ramp = 1.0
                gen_total = gen_total + ramp * gan_weight * adapt_w * g_loss
                g_loss_val = float(g_loss.item())

            scaled_loss = gen_total / grad_accum_steps

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        # Step generator optimizer at accumulation boundary
        if is_accum_step:
            if scaler is not None:
                scaler.step(optimizer)
            else:
                optimizer.step()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # ---- Discriminator step (at accumulation boundary only) ----
        d_loss_val = 0.0
        if disc_active and is_accum_step:
            with torch.cuda.amp.autocast(dtype=amp_dtype, enabled=use_amp):
                with torch.no_grad():
                    recon_01_det = _denormalize(loss_output.recon.detach(), device)
                    x_01_det = _denormalize(x.detach(), device)
                logits_real = discriminator(x_01_det)
                logits_fake = discriminator(recon_01_det)
                d_loss = hinge_loss_d(logits_real, logits_fake)

            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(opt_disc)
            else:
                d_loss.backward()
                opt_disc.step()
            opt_disc.zero_grad()
            d_loss_val = float(d_loss.item())

        # Update scaler once after both optimizer steps
        if is_accum_step and scaler is not None:
            scaler.update()

        # Statistics
        rec_loss_val = float(loss_output.rec_loss.item())
        aux_loss_val = float(loss_output.aux_loss.item())
        aux_loss_name = loss_output.aux_loss_name

        total_rec_loss += rec_loss_val
        total_aux_loss += aux_loss_val
        total_percep_loss += p_loss_val
        total_g_loss += g_loss_val
        total_d_loss += d_loss_val
        total_active_codes += loss_output.active_codes
        num_batches += 1
        global_step += 1

        if print_every > 0 and (batch_idx % print_every == 0):
            msg = f"{prefix} [{batch_idx}/{num_iters}]  rec={rec_loss_val:.4f}  active={loss_output.active_codes}"
            if aux_loss_name != "none":
                msg += f"  {aux_loss_name[:3]}={aux_loss_val:.4f}"
            if loss_output.perplexity is not None:
                msg += f"  ppl={loss_output.perplexity:.1f}"
            if use_percep:
                msg += f"  percep={p_loss_val:.4f}"
            if disc_active:
                msg += f"  g={g_loss_val:.4f}  d={d_loss_val:.4f}"
            print(msg, flush=True)

        if csv_logger is not None and csv_log_every > 0 and (global_step % csv_log_every == 0):
            lr = optimizer.param_groups[0].get("lr", None)
            log_dict = {
                "epoch": epoch,
                "global_step": global_step,
                "batch_idx": batch_idx,
                "rec_loss": rec_loss_val,
                "total_loss": float(gen_total.item()),
                "active_codes": loss_output.active_codes,
                "perplexity": loss_output.perplexity,
                "lr": lr,
            }
            if aux_loss_name != "none":
                log_dict[aux_loss_name] = aux_loss_val
            if use_percep:
                log_dict["perceptual_loss"] = p_loss_val
            if use_gan:
                log_dict["g_loss"] = g_loss_val
                log_dict["d_loss"] = d_loss_val
            csv_logger.log(log_dict)

        if wandb_run is not None and csv_log_every > 0 and (global_step % csv_log_every == 0):
            wb_dict = {
                "train/rec_loss": rec_loss_val,
                "train/total_loss": float(gen_total.item()),
                "train/active_codes": loss_output.active_codes,
                "train/lr": optimizer.param_groups[0].get("lr", 0),
            }
            if loss_output.perplexity is not None:
                wb_dict["train/perplexity"] = loss_output.perplexity
            if aux_loss_name != "none":
                wb_dict[f"train/{aux_loss_name}"] = aux_loss_val
            if use_percep:
                wb_dict["train/perceptual_loss"] = p_loss_val
            if use_gan:
                wb_dict["train/g_loss"] = g_loss_val
                wb_dict["train/d_loss"] = d_loss_val
            wandb_run.log(wb_dict, step=global_step)

    metrics = {
        "rec_loss": total_rec_loss / num_batches,
        "active_codes": total_active_codes / num_batches,
    }
    if aux_loss_name != "none":
        metrics[aux_loss_name] = total_aux_loss / num_batches
    if use_percep:
        metrics["perceptual_loss"] = total_percep_loss / num_batches
    if use_gan:
        metrics["g_loss"] = total_g_loss / num_batches
        metrics["d_loss"] = total_d_loss / num_batches

    return metrics, global_step


@torch.no_grad()
def evaluate(
    model,
    model_config,
    val_loader,
    device,
    *,
    max_samples: Optional[int] = None,
    run_name: str = "",
    compute_fid: bool = False,
    fid_samples: Optional[int] = None,
    fid_calculator: Optional[FIDCalculator] = None,
    lpips_metric: Optional[LPIPSMetric] = None,
    codebook_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Returns dict with:
        - val_rec_loss: Mean reconstruction loss
        - val_total_loss: Mean total loss (rec + aux)
        - val_active_codes: Number of unique codes used
        - val_codebook_util: Percentage of codebook used
        - val_perplexity: Code usage perplexity (K_eff)
        - val_entropy_bits: Empirical marginal entropy of assignments (bits)
        - val_bits_per_token: Bits per token under i.i.d. marginal
        - val_psnr: Peak Signal-to-Noise Ratio
        - val_ssim: Structural Similarity Index
        - val_lpips: Learned Perceptual Image Patch Similarity
        - val_aux_loss: Auxiliary loss (commit/entropy/vq)
        - val_rfid: Reconstruction FID
    """
    model.eval()
    
    total_rec_loss = 0.0
    total_aux_loss = 0.0
    total_total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_lpips_samples = 0  # sample-weighted average (last batch may be smaller)
    lpips_failed = False
    all_indices = []
    num_batches = 0
    aux_loss_name = "aux_loss"
    samples_seen = 0
    
    # Initialize LPIPS metric if not provided
    if lpips_metric is None:
        try:
            lpips_metric = LPIPSMetric(device=device)
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS: {e}")
            lpips_metric = None
    
    # Denormalization for metrics (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    for x, _ in val_loader:
        if max_samples and samples_seen >= max_samples:
            break
            
        x = x.to(device)
        samples_seen += x.size(0)
        
        # Use model-specific loss computation
        loss_output: LossOutput = model_config.compute_loss(model, x)
        
        total_rec_loss += float(loss_output.rec_loss.item())
        total_aux_loss += float(loss_output.aux_loss.item())
        total_total_loss += float(loss_output.total_loss.item())
        aux_loss_name = loss_output.aux_loss_name
        num_batches += 1
        
        # Get reconstructions and indices
        outputs = model(x)
        recon = outputs[0]
        if len(outputs) >= 2:
            indices = outputs[1]
            all_indices.append(indices.cpu())
        
        # Denormalize for image quality metrics
        x_denorm = (x * std + mean).clamp(0, 1)
        recon_denorm = (recon * std + mean).clamp(0, 1)
        
        # Compute PSNR and SSIM
        total_psnr += compute_psnr(x_denorm, recon_denorm)
        total_ssim += compute_ssim(x_denorm, recon_denorm)
        
        # Compute LPIPS (sample-weighted average so last batch is not over-weighted)
        if lpips_metric is not None and not lpips_failed:
            try:
                batch_lpips = lpips_metric.compute(x_denorm, recon_denorm)
                total_lpips += batch_lpips * x.size(0)
                total_lpips_samples += x.size(0)
            except Exception:
                lpips_failed = True
    
    # Compute global codebook metrics
    if all_indices:
        # Detect LGQ by checking if model has the LGQ quantizer (model.quantize.K exists)
        sample_idx = all_indices[0]
        is_lmb = hasattr(model, 'quantize') and hasattr(model.quantize, 'K')
        
        if is_lmb:
            # Check if using flattened mode
            flatten_channels = getattr(model.quantize, 'flatten_channels', False)
            # Use the quantizer's declared sizes — inferring num_bins from a single
            # batch's max index undercounts when no batch happens to hit the top bin,
            # which made codebook_util exceed 100%.
            num_bins = int(model.quantize.K)

            if flatten_channels:
                # FLATTENED MODE: indices are [B, T] with values 0..num_bins-1
                # Each index represents a codebook entry for the entire C-dimensional vector
                global_active_codes = int(torch.cat([idx.view(-1) for idx in all_indices]).unique().numel())
                effective_codebook_size = num_bins

                # Compute perplexity
                all_codes = torch.cat([idx.view(-1) for idx in all_indices])
                counts = torch.bincount(all_codes.long(), minlength=num_bins)
                counts = counts[counts > 0].float()
                probs = counts / counts.sum()
                global_perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())

                codebook_size = num_bins
            else:
                # PER-CHANNEL MODE: indices are [B, T, C] with values 0..num_bins-1
                # Need to count unique (channel, bin) pairs
                C = int(model.quantize.C)
                
                all_unique_pairs = set()
                for idx in all_indices:
                    B, T, C_dim = idx.shape
                    for c in range(C_dim):
                        # For each channel, get the bin indices used
                        channel_bins = idx[:, :, c].view(-1)  # [B*T]
                        # Create unique code = c * num_bins + bin_idx
                        unique_codes = (c * num_bins + channel_bins).unique().tolist()
                        all_unique_pairs.update(unique_codes)
                
                global_active_codes = len(all_unique_pairs)
                effective_codebook_size = C * num_bins  # e.g., 128 * 128 = 16384
                
                # Compute perplexity across all (channel, bin) pairs
                all_codes = []
                for idx in all_indices:
                    B, T, C_dim = idx.shape
                    for c in range(C_dim):
                        channel_bins = idx[:, :, c].view(-1)
                        codes = c * num_bins + channel_bins
                        all_codes.append(codes)
                all_codes_cat = torch.cat(all_codes)
                counts = torch.bincount(all_codes_cat.long(), minlength=effective_codebook_size)
                counts = counts[counts > 0].float()
                probs = counts / counts.sum()
                global_perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
                
                # Override codebook_size for utilization calculation
                if codebook_size is None or codebook_size == num_bins:
                    codebook_size = effective_codebook_size
        else:
            # Standard VQ/FSQ/SIM_VQ: indices are [B, T] with values 0..codebook_size-1
            all_indices_cat = torch.cat([idx.view(-1) for idx in all_indices])
            global_active_codes = int(all_indices_cat.unique().numel())
            
            # If codebook_size not provided, try to infer from model
            if codebook_size is None or codebook_size == 0:
                if hasattr(model, 'quantize') and hasattr(model.quantize, 'codebook_size'):
                    codebook_size = model.quantize.codebook_size
                elif hasattr(model, 'num_codes'):
                    codebook_size = model.num_codes
                else:
                    # Fallback: use max index + 1 as codebook size estimate
                    max_idx = int(all_indices_cat.max().item()) + 1
                    codebook_size = max_idx
            
            # Compute global perplexity
            counts = torch.bincount(all_indices_cat.long(), minlength=max(1, codebook_size))
            counts = counts[counts > 0].float()
            probs = counts / counts.sum()
            global_perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
        
        # Compute codebook utilization
        if codebook_size is not None and codebook_size > 0:
            codebook_util = 100.0 * global_active_codes / codebook_size
        else:
            codebook_util = 0.0
    else:
        global_active_codes = 0
        global_perplexity = 0.0
        codebook_util = 0.0

    # Entropy-based rate: H = log2(perplexity), bits per token = H
    val_entropy_bits = math.log2(max(global_perplexity, 1.0))
    val_bits_per_token = val_entropy_bits
    # Tokens per image: encoder does 16x downsample, so (H/16)*(W/16) spatial tokens
    # For 256x256 input: 16x16 = 256 tokens per image
    tokens_per_image = (model.enc.stem.weight.shape[0] > 0) and 256  # default
    if all_indices and len(all_indices) > 0:
        # Indices may be returned flattened [B, T] or spatial [B, H, W] —
        # multiply all non-batch dims to get total tokens per image.
        idx_shape = all_indices[0].shape
        if len(idx_shape) >= 2:
            tokens_per_image = math.prod(idx_shape[1:])
    val_bits_per_image = val_bits_per_token * tokens_per_image

    metrics = {
        "val_rec_loss": total_rec_loss / max(num_batches, 1),
        "val_total_loss": total_total_loss / max(num_batches, 1),
        "val_active_codes": global_active_codes,
        "val_codebook_util": codebook_util,
        "val_perplexity": global_perplexity,
        "val_entropy_bits": val_entropy_bits,
        "val_bits_per_token": val_bits_per_token,
        "val_bits_per_image": val_bits_per_image,
        "val_tokens_per_image": tokens_per_image,
        "val_psnr": total_psnr / max(num_batches, 1),
        "val_ssim": total_ssim / max(num_batches, 1),
        "val_lpips": float("nan") if (lpips_failed or total_lpips_samples == 0 or lpips_metric is None) else total_lpips / total_lpips_samples,
    }
    
    if aux_loss_name != "none":
        metrics[f"val_{aux_loss_name}"] = total_aux_loss / max(num_batches, 1)
    
    # Compute rFID (reconstruction FID)
    if compute_fid:
        try:
            if fid_calculator is None:
                fid_calculator = FIDCalculator(device=device)
            
            rfid = fid_calculator.compute_rfid(
                model=model,
                dataloader=val_loader,
                max_samples=fid_samples or max_samples,
                show_progress=True,
            )
            metrics["val_rfid"] = rfid
        except Exception as e:
            print(f"Warning: Failed to compute rFID: {e}")
            metrics["val_rfid"] = float("nan")
    
    return metrics


def main():
    # First pass: get model type to add model-specific args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model", type=str, default="fsq")
    pre_args, _ = pre_parser.parse_known_args()
    
    # Create full parser with common args
    parser = create_common_parser()
    
    # Add model-specific args
    model_type = pre_args.model
    if model_type in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[model_type]
        model_config.add_model_args(parser)
    
    args = parser.parse_args()
    
    # Get model config
    model_config = MODEL_CONFIGS[args.model]
    
    # Generate run name if not provided
    if args.run_name is None:
        run_name = generate_run_name(args)
    else:
        run_name = args.run_name
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_root, args.model, run_name)
    
    # DDP init
    ddp = is_dist_env()
    local_rank = get_local_rank()
    world_size = get_world_size()
    rank = get_rank()
    if ddp:
        torch.cuda.set_device(local_rank)
        # Bump PG timeout past the default 10 min: rank 0 runs FID/LPIPS on
        # fid_samples images while other ranks wait at the epoch-end barrier,
        # and that can exceed 10 min on large val sets.
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=1))
        device = f"cuda:{local_rank}"
        rprint(f"DDP: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    else:
        device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
        rprint(f"Using device: {device}")

    # cuDNN disabled by default: module cuda toolkit (11.x) lacks H100 sm_90 kernels for
    # every bf16 conv shape, producing "GET/FIND was unable to find an engine"
    # on the encoder stem. Native ATen conv works under bf16. Override with
    # VQ_CUDNN_ENABLED=1 to test cuDNN-on (sets benchmark=True for algo selection).
    if torch.cuda.is_available():
        cudnn_on = os.environ.get("VQ_CUDNN_ENABLED", "0") == "1"
        torch.backends.cudnn.enabled = cudnn_on
        if cudnn_on:
            torch.backends.cudnn.benchmark = True
        rprint(f"cuDNN: enabled={cudnn_on} benchmark={torch.backends.cudnn.benchmark}")

    # Set random seed (offset by rank so DDP ranks get different augmentation noise)
    torch.manual_seed(args.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + rank)
    
    # Create output directory (rank 0 only)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    config_data = vars(args).copy()
    config_data["run_name"] = run_name
    config_data["output_dir"] = str(output_dir)
    config_data["world_size"] = world_size
    if is_main():
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir.mkdir(exist_ok=True)
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"\n{'='*70}")
        print(f"Run: {run_name}")
        print(f"Model: {args.model.upper()}")
        print(f"Output: {output_dir}")
        print(f"World size: {world_size}")
        print(f"{'='*70}")
        print(model_config.get_model_info(args))
    if ddp:
        dist.barrier()
    
    # Data transforms (same for all models)
    image_size = args.image_size
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_root = os.path.expanduser(args.data_root)

    # Load training dataset
    dataset_name = getattr(args, "dataset", "imagenet")
    if dataset_name == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform
        )
    elif dataset_name == "imagenet100":
        synsets = get_imagenet100_synsets(data_root)
        train_dataset = ImageNet100Folder(
            root=os.path.join(data_root, "train"),
            transform=transform,
            synsets=synsets,
        )
    else:
        train_dataset = SafeImageFolder(
            root=os.path.join(data_root, "train"),
            transform=transform
        )
    # Optional fixed-seed subset for small-sample ablations (deterministic across ranks).
    max_train = getattr(args, "max_train_samples", 0) or 0
    if max_train > 0 and max_train < len(train_dataset):
        g = torch.Generator().manual_seed(args.seed)
        subset_idx = torch.randperm(len(train_dataset), generator=g)[:max_train].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, subset_idx)
        rprint(f"Subset training set to {len(train_dataset)} images (--max-train-samples={max_train}, seed={args.seed})")

    train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed) if ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.num_workers > 0,
        persistent_workers=args.num_workers > 0,
        drop_last=ddp,
    )

    rprint(f"Loaded {len(train_dataset)} training images (per-rank batch={args.batch_size}, global={args.batch_size * world_size})")
    
    # Load validation dataset for evaluation (rank 0 only runs eval)
    val_loader = None
    if not args.no_eval and is_main():
        eval_batch_size = args.eval_batch_size or args.batch_size

        if dataset_name == "cifar10":
            val_dataset = datasets.CIFAR10(
                root=data_root, train=False, download=True, transform=transform
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=args.num_workers > 0,
                persistent_workers=args.num_workers > 0,
            )
            print(f"Loaded {len(val_dataset)} CIFAR-10 test images for evaluation")
        else:
            # Try val folder first, then test folder
            for split_name, split_folder in [("val", "val"), ("test", "test")]:
                split_path = os.path.join(data_root, split_folder)
                if os.path.exists(split_path):
                    try:
                        # First try ImageFolder (for datasets with class subdirectories)
                        if dataset_name == "imagenet100":
                            val_dataset = ImageNet100Folder(root=split_path, transform=transform, synsets=synsets)
                        else:
                            val_dataset = SafeImageFolder(root=split_path, transform=transform)
                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=eval_batch_size,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=args.num_workers > 0,
                            persistent_workers=args.num_workers > 0,
                        )
                        print(f"Loaded {len(val_dataset)} {split_name} images for evaluation (ImageFolder format)")
                        break
                    except (FileNotFoundError, RuntimeError):
                        # ImageFolder requires class subfolders - try FlatImageDataset instead
                        try:
                            val_dataset = FlatImageDataset(root=split_path, transform=transform)
                            val_loader = DataLoader(
                                val_dataset,
                                batch_size=eval_batch_size,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=args.num_workers > 0,
                                persistent_workers=args.num_workers > 0,
                            )
                            print(f"Loaded {len(val_dataset)} {split_name} images for evaluation (flat directory)")
                            break
                        except RuntimeError as e:
                            print(f"Note: Could not load {split_path}: {e}")
                            continue

        if val_loader is None:
            print("Warning: No valid evaluation dataset found, skipping evaluation")
    
    # Create model config (same architecture for all)
    model_cfg = ModelConfig(
        base_ch=args.dim // 2,  # base_ch (dim/2 for comparable capacity)
        embedding_dim=args.embedding_dim,
        legacy_decoder=getattr(args, 'legacy_decoder', False),
    )
    
    # Create model using unified architecture with model-specific quantizer
    model = model_config.create_model(args, model_cfg).to(device)

    # Load pre-computed codebook initialisation from Phase 0 (before DDP wrap)
    codebook_init_path = getattr(args, "codebook_init", None)
    if codebook_init_path is not None:
        import numpy as np
        codebook_np = np.load(codebook_init_path)
        codebook_tensor = torch.from_numpy(codebook_np).float().to(device)
        if hasattr(model, "quantize") and hasattr(model.quantize, "centers"):
            K_init, C_init = codebook_tensor.shape
            K_model = model.quantize.centers.shape[0]
            # unwrap not needed here (DDP wrap happens below)
            if K_init != K_model:
                print(f"WARNING: codebook init has {K_init} codes but model expects {K_model}; truncating/padding")
                if K_init > K_model:
                    codebook_tensor = codebook_tensor[:K_model]
                else:
                    pad = torch.randn(K_model - K_init, C_init, device=device) * 0.01
                    codebook_tensor = torch.cat([codebook_tensor, pad], dim=0)
            model.quantize.centers.data.copy_(codebook_tensor)
            model.quantize._inited.fill_(1)  # skip lazy init
            print(f"Loaded codebook init from {codebook_init_path} ({codebook_tensor.shape})")
        else:
            print(f"WARNING: --codebook-init specified but model has no centers attribute; ignoring")

    # Print parameter breakdown (rank 0)
    if is_main():
        param_counts = count_model_parameters(model)
        print(f"\nArchitecture: unified Encoder/Decoder")
        print(f"  Encoder: {param_counts['encoder']:,}")
        print(f"  Decoder: {param_counts['decoder']:,}")
        print(f"  Quantizer: {param_counts['quantizer']:,}")
        if param_counts['projections'] > 0:
            print(f"  Projections: {param_counts['projections']:,}")
        print(f"  Total: {param_counts['total']:,}")
        print()

    # Trigger data-dependent codebook init on rank 0 before DDP wrap.
    # Quantizers with a `_maybe_init_from_batch` path (e.g. LGQ) populate
    # `self.centers` from the first forward-pass batch. Under DDP each rank
    # would otherwise init against its own shard with its own per-rank seed,
    # leaving the four codebooks permanently diverged — DDP syncs gradients
    # but not post-init parameters. Running one forward on rank 0 first, then
    # wrapping, lets DDP's init-time parameter+buffer broadcast carry a single
    # codebook (and `_inited=1`) to every rank.
    if ddp and _quantizer_needs_lazy_init(model):
        if is_main():
            with torch.no_grad():
                init_iter = iter(train_loader)
                init_x = next(init_iter)[0].to(device)
                _ = model(init_x)
                del init_iter, init_x
        dist.barrier()

    # Wrap model in DDP
    if ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Optimizer (same for all models) — params from DDP wrapper OK
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Perceptual + adversarial loss setup
    perceptual_loss_fn = None
    discriminator = None
    opt_disc = None
    use_percep = getattr(args, "perceptual_weight", 0.0) > 0
    use_gan = getattr(args, "gan_weight", 0.0) > 0

    if use_percep:
        perceptual_loss_fn = PerceptualLoss(device=device)
        print(f"Perceptual loss enabled (weight={args.perceptual_weight})")

    if use_gan:
        disc_n_layers = getattr(args, "disc_n_layers", 3)
        discriminator = PatchGANDiscriminator(
            in_channels=3, ndf=64, n_layers=disc_n_layers
        ).to(device)
        opt_disc = torch.optim.AdamW(
            discriminator.parameters(), lr=getattr(args, "disc_lr", args.lr)
        )
        if ddp:
            discriminator = DDP(discriminator, device_ids=[local_rank], find_unused_parameters=False)
        d_params = sum(p.numel() for p in unwrap(discriminator).parameters())
        rprint(f"PatchGAN discriminator enabled (weight={args.gan_weight}, "
               f"start_step={args.disc_start}, params={d_params:,})")

    # Mixed precision setup
    amp_dtype = None
    scaler = None
    if args.amp:
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            print("AMP enabled with bfloat16")
        else:
            amp_dtype = torch.float16
            scaler = torch.cuda.amp.GradScaler()
            print("AMP enabled with float16 + GradScaler")

    if args.grad_accum_steps > 1:
        rprint(f"Gradient accumulation: {args.grad_accum_steps} steps "
               f"(effective batch size = {args.batch_size * args.grad_accum_steps * world_size})")

    # LR scheduler: linear warmup + optional cosine decay
    steps_per_epoch = max(1, len(train_loader) // max(1, args.grad_accum_steps))
    total_opt_steps = steps_per_epoch * args.epochs
    warmup = max(0, args.warmup_steps)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        if args.lr_schedule == "cosine":
            progress = (step - warmup) / max(1, total_opt_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_frac = args.min_lr / args.lr if args.lr > 0 else 0.0
            return min_frac + (1.0 - min_frac) * cos
        return 1.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rprint(f"LR schedule: warmup={warmup} steps, schedule={args.lr_schedule}, total_opt_steps={total_opt_steps}")

    # Training loop
    best_val_loss = float("inf")
    train_history = []
    global_step = 0
    start_epoch = 1
    
    # Resume from checkpoint if specified
    if args.resume:
        if args.resume.lower() == "latest":
            checkpoint_path = checkpoint_dir / "latest_model.pt"
        elif args.resume.lower() == "best":
            checkpoint_path = checkpoint_dir / "best_model.pt"
        else:
            checkpoint_path = Path(args.resume)
        
        if checkpoint_path.exists():
            rprint(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            unwrap(model).load_state_dict(checkpoint["model_state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint.get("epoch", 1) + 1
            global_step = checkpoint.get("global_step", 0)
            train_history = checkpoint.get("train_history", [])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            # Restore discriminator state if available
            if discriminator is not None and "disc_state_dict" in checkpoint:
                unwrap(discriminator).load_state_dict(checkpoint["disc_state_dict"])
                opt_disc.load_state_dict(checkpoint["opt_disc_state_dict"])
                rprint("  Restored discriminator state")
            # Fast-forward scheduler to resume step
            for _ in range(global_step):
                lr_scheduler.step()
            rprint(f"Resumed from epoch {start_epoch - 1}, global_step={global_step}")
        else:
            rprint(f"Warning: Checkpoint {checkpoint_path} not found, starting from scratch")
    
    csv_logger = CsvMetricLogger(output_dir / "train_metrics.csv") if is_main() else None
    eval_csv_logger = CsvMetricLogger(output_dir / "eval_metrics.csv") if (val_loader and is_main()) else None

    # Wandb setup (rank 0 only)
    wandb_run = None
    if getattr(args, "wandb", False) and is_main():
        try:
            import wandb
            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", "vector-quantize"),
                entity=getattr(args, "wandb_entity", None),
                name=run_name,
                config=config_data,
                dir=str(output_dir),
                resume="allow",
            )
            print(f"W&B logging enabled: {wandb_run.url}")
        except Exception as e:
            print(f"Warning: Could not initialize W&B: {e}")
            wandb_run = None
    
    # Metrics calculators (lazily initialized on first use)
    fid_calculator = None
    lpips_metric = None
    
    # Get codebook size from model for utilization calculation
    codebook_size = None
    
    _m = unwrap(model)
    if hasattr(_m, 'quantize'):
        if hasattr(_m.quantize, 'codebook_size'):
            codebook_size = _m.quantize.codebook_size
        elif hasattr(_m, 'num_codes'):
            codebook_size = _m.num_codes
    if codebook_size is None:
        codebook_size = getattr(_m, 'codebook_size', None)
    if codebook_size is None and hasattr(_m, 'quantizer'):
        codebook_size = getattr(_m.quantizer, 'codebook_size', None)
    
    if val_loader is not None:
        print(f"Evaluation metrics: rec_loss, PSNR, SSIM, LPIPS, rFID, codebook_util")
    
    for epoch in range(start_epoch, args.epochs + 1):
        rprint(f"\n[{run_name}] Epoch {epoch}/{args.epochs}")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Model-specific epoch updates (e.g., temperature annealing for LGQ)
        if hasattr(model_config, 'update_model_for_epoch'):
            info = model_config.update_model_for_epoch(unwrap(model), args, epoch, args.epochs)
            if info:
                rprint(f"  {info}")
        
        # Training
        train_metrics, global_step = train_epoch(
            model,
            model_config,
            train_loader,
            optimizer,
            device,
            epoch=epoch,
            global_step=global_step,
            csv_logger=csv_logger,
            csv_log_every=args.csv_log_every,
            print_every=args.print_every,
            max_steps=args.max_steps,
            run_name=run_name,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            amp_dtype=amp_dtype,
            perceptual_loss_fn=perceptual_loss_fn,
            discriminator=discriminator,
            opt_disc=opt_disc,
            perceptual_weight=getattr(args, "perceptual_weight", 0.0),
            gan_weight=getattr(args, "gan_weight", 0.0),
            l1_weight=getattr(args, "l1_weight", 1.0),
            disc_start=getattr(args, "disc_start", 10000),
            disc_warmup_steps=getattr(args, "disc_warmup_steps", 0),
            lr_scheduler=lr_scheduler,
            wandb_run=wandb_run,
            lfq_entropy_warmup_steps=getattr(args, "lfq_entropy_warmup_steps", 0),
            lfq_entropy_warmup_factor=getattr(args, "lfq_entropy_warmup_factor", 1.0),
        )
        train_metrics["epoch"] = epoch
        
        # Evaluation (rank 0 only; other ranks wait at barrier after)
        eval_metrics = {}
        if val_loader is not None and is_main():
            if fid_calculator is None:
                fid_calculator = FIDCalculator(device=device)
            if lpips_metric is None:
                try:
                    lpips_metric = LPIPSMetric(device=device)
                except Exception as e:
                    print(f"Warning: LPIPS not available: {e}")

            eval_metrics = evaluate(
                unwrap(model),
                model_config,
                val_loader,
                device,
                max_samples=args.eval_samples,
                run_name=run_name,
                compute_fid=not args.no_rfid,
                fid_samples=args.fid_samples,
                fid_calculator=fid_calculator,
                lpips_metric=lpips_metric,
                codebook_size=codebook_size,
            )
            
        if val_loader is not None and is_main():
            # Log evaluation metrics
            if eval_csv_logger:
                eval_log = {"epoch": epoch, "global_step": global_step}
                eval_log.update(eval_metrics)
                eval_csv_logger.log(eval_log)
            
            # Log eval metrics to wandb
            if wandb_run is not None:
                wb_eval = {f"val/{k.removeprefix('val_')}": v
                           for k, v in eval_metrics.items()
                           if isinstance(v, (int, float))}
                wb_eval["epoch"] = epoch
                wandb_run.log(wb_eval, step=global_step)

            # Print evaluation summary
            print(f"  Val: rec_loss={eval_metrics['val_rec_loss']:.4f}, "
                  f"PSNR={eval_metrics['val_psnr']:.2f}dB, "
                  f"SSIM={eval_metrics['val_ssim']:.4f}, "
                  f"LPIPS={eval_metrics['val_lpips']:.4f}")
            print(f"       rFID={eval_metrics.get('val_rfid', float('nan')):.2f}, "
                  f"codebook_util={eval_metrics['val_codebook_util']:.1f}%, "
                  f"perplexity={eval_metrics['val_perplexity']:.1f}, "
                  f"entropy={eval_metrics.get('val_entropy_bits', 0):.2f} bits, "
                  f"bits/token={eval_metrics.get('val_bits_per_token', 0):.2f}, "
                  f"bits/image={eval_metrics.get('val_bits_per_image', 0):.1f}")
        
        # Combine metrics
        all_metrics = {**train_metrics, **eval_metrics}
        train_history.append(all_metrics)

        # Save checkpoint (rank 0 only)
        if is_main():
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": unwrap(model).state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "train_history": train_history,
                "config": config_data,
                "model_type": args.model,
                "run_name": run_name,
            }
            if discriminator is not None:
                checkpoint["disc_state_dict"] = unwrap(discriminator).state_dict()
                checkpoint["opt_disc_state_dict"] = opt_disc.state_dict()

            if not getattr(args, "keep_only_best_latest", False):
                torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt")

            if epoch == 1 and getattr(args, "save_initial_centers", False) and args.model in ("lgq", "lmb"):
                for key, value in unwrap(model).state_dict().items():
                    if "quantize.centers" in key:
                        analysis_path = checkpoint_dir / "analysis_epoch001_centers.pt"
                        flatten = config_data.get("flatten_channels", False)
                        torch.save({
                            "epoch": 1,
                            "centers": value.cpu(),
                            "flatten_channels": flatten,
                        }, analysis_path)
                        print(f"  Saved initial centers for analysis: {analysis_path.name}")
                        break

            checkpoint["best_val_loss"] = best_val_loss
            checkpoint["global_step"] = global_step
            torch.save(checkpoint, checkpoint_dir / "latest_model.pt")

            current_loss = eval_metrics.get("val_rec_loss", train_metrics["rec_loss"])
            if current_loss < best_val_loss:
                best_val_loss = current_loss
                torch.save(checkpoint, checkpoint_dir / "best_model.pt")
                loss_type = "val_rec_loss" if "val_rec_loss" in eval_metrics else "train_rec_loss"
                print(f"  Saved best model ({loss_type}: {best_val_loss:.4f})")

            with open(output_dir / "train_history.json", "w") as f:
                json.dump(train_history, f, indent=2)

        # Sync all ranks before next epoch
        if ddp:
            dist.barrier()

    if wandb_run is not None:
        wandb_run.finish()

    rprint(f"\n[{run_name}] Training completed!")
    rprint(f"Results saved to: {output_dir}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    # Only set spawn for single-process runs; torchrun manages its own processes.
    if not is_dist_env():
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
    main()

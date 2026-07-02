#!/usr/bin/env python3
"""
Standalone evaluation script for trained quantization models.

Evaluates a checkpoint on the validation/test set and reports metrics.

Usage:
    python scripts/evaluate.py --checkpoint results/fsq/checkpoints/best_model.pt
    python scripts/evaluate.py --checkpoint results/vq/checkpoints/best_model.pt --save-samples
"""

import argparse
import faulthandler
import json
import math
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm
from typing import Optional, Dict

faulthandler.enable(file=sys.stderr, all_threads=True)

import torch
from torch.utils.data import DataLoader
TORCHVISION_AVAILABLE = False
try:
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

from PIL import Image
import numpy as np

from configs import MODEL_CONFIGS
from configs.base import LossOutput
from quantization.model import ModelConfig, UnifiedAutoEncoder, QuantizerConfig
from scripts.evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric
from torch.utils.data import Dataset
from pathlib import Path

# Copy FlatImageDataset here to avoid importing from train.py (which needs torchvision)
class FlatImageDataset(Dataset):
    """Simple image dataset that loads all images from a directory."""
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        root_path = Path(root)
        for ext in self.EXTENSIONS:
            self.image_paths.extend(root_path.glob(f'*{ext}'))
            self.image_paths.extend(root_path.glob(f'*{ext.upper()}'))
            self.image_paths.extend(root_path.glob(f'**/*{ext}'))
            self.image_paths.extend(root_path.glob(f'**/*{ext.upper()}'))
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
        return img, 0

# Lazy import for FID (requires torchvision)
FIDCalculator = None
try:
    from scripts.evaluation.fid import FIDCalculator
except ImportError:
    pass


def load_model_from_checkpoint(checkpoint_path: str, device: str) -> tuple:
    """Load model and config from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint["config"]
    model_type = checkpoint.get("model_type", config.get("model"))
    
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_config_cls = MODEL_CONFIGS[model_type]
    
    # Reconstruct args namespace
    args = argparse.Namespace(**config)
    
    # Create model config
    model_cfg = ModelConfig(
        base_ch=config.get("dim", 256) // 2,
        embedding_dim=config.get("embedding_dim", 256),
    )
    
    # Create model
    model = model_config_cls.create_model(args, model_cfg)
    
    # Handle transposed quantizer weights for flattened LMB models
    state_dict = checkpoint["model_state_dict"].copy()
    if "quantize.centers" in state_dict:
        centers = state_dict["quantize.centers"]
        # Check if dimensions are transposed (checkpoint has [K, C] but model expects [C, K])
        if model.quantize.centers.shape != centers.shape:
            if centers.shape == (centers.shape[0], model.quantize.centers.shape[1]) and \
               centers.shape[1] == model.quantize.centers.shape[0]:
                print(f"Transposing quantize.centers from {centers.shape} to {model.quantize.centers.shape}")
                state_dict["quantize.centers"] = centers.t()
        if "quantize._dbg_usage" in state_dict:
            usage = state_dict["quantize._dbg_usage"]
            if model.quantize._dbg_usage.shape != usage.shape:
                # _dbg_usage should match centers shape [C, K]
                if usage.shape[0] == model.quantize.centers.shape[0] * model.quantize.centers.shape[1]:
                    # Flattened, reshape to [C, K]
                    print(f"Reshaping quantize._dbg_usage from {usage.shape} to {model.quantize._dbg_usage.shape}")
                    state_dict["quantize._dbg_usage"] = usage.view(model.quantize._dbg_usage.shape)
                elif usage.shape == (model.quantize.centers.shape[1], model.quantize.centers.shape[0]):
                    # Transposed
                    state_dict["quantize._dbg_usage"] = usage.t()
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    return model, model_config_cls, args, checkpoint


@torch.no_grad()
def evaluate(
    model,
    model_config,
    val_loader,
    device,
    *,
    max_samples: Optional[int] = None,
    save_samples: bool = False,
    output_dir: Optional[Path] = None,
    num_save: int = 16,
    compute_fid: bool = True,
    fid_samples: Optional[int] = None,
    fid_calculator: Optional[FIDCalculator] = None,
    codebook_size: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.
    
    Returns dict with metrics: rec_loss, PSNR, SSIM, LPIPS, rFID, codebook_util.
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
    
    # Initialize LPIPS metric
    try:
        lpips_metric = LPIPSMetric(device=device)
    except Exception as e:
        print(f"Warning: LPIPS not available: {e}")
        lpips_metric = None
    
    # Denormalization for metrics (ImageNet normalization)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    # For saving samples
    saved_originals = []
    saved_recons = []
    
    pbar = tqdm(val_loader, desc="Evaluating")
    for x, _ in pbar:
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
            except Exception as e:
                if num_batches == 1:
                    print(f"Warning: LPIPS failed: {e}")
                lpips_failed = True
        
        # Save samples
        if save_samples and len(saved_originals) < num_save:
            n_to_save = min(num_save - len(saved_originals), x.size(0))
            saved_originals.append(x[:n_to_save].cpu())
            saved_recons.append(recon[:n_to_save].cpu())
        
        # Update progress
        pbar.set_postfix({
            "rec": f"{loss_output.rec_loss.item():.4f}",
            "PSNR": f"{total_psnr / num_batches:.1f}",
        })
    
    # Compute global active codes
    # Detect LMB by checking if model has the LMB quantizer (model.quantize.K exists)
    # LMB indices are [B, T, C] where C=embedding_dim and values are 0..num_bins-1
    # FSQ/VQ indices are [B, H, W] or [B, T] with values 0..codebook_size-1
    sample_idx = all_indices[0]
    is_lmb = hasattr(model, 'quantize') and (
        hasattr(model.quantize, 'K') or hasattr(model.quantize, 'levels')
    )
    is_lmb_fair = hasattr(model, 'quantize') and hasattr(model.quantize, 'levels')
    
    if is_lmb and is_lmb_fair:
        # LMB perchannel_fair: indices are [B, T, C] with per-channel bin indices
        # Need to compute proper product indices for utilization
        levels = model.quantize.levels
        C = sample_idx.shape[2]
        codebook_size = model.quantize.codebook_size
        
        # Compute strides for product index: [L1*L2*...*L_{C-1}, L2*..., ..., 1]
        strides = []
        stride = 1
        for c in range(C - 1, -1, -1):
            strides.insert(0, stride)
            stride *= levels[c]
        strides = torch.tensor(strides, device=sample_idx.device, dtype=torch.long)
        
        # Compute product indices for all batches
        all_product_indices = []
        for idx in all_indices:
            # idx shape: [B, T, C] with per-channel bin indices
            # product_idx = sum(idx[c] * strides[c]) for proper product space mapping
            product_idx = (idx.long() * strides.view(1, 1, -1)).sum(dim=-1)  # [B, T]
            all_product_indices.append(product_idx.view(-1))
        
        all_codes_cat = torch.cat(all_product_indices)
        global_active_codes = int(all_codes_cat.unique().numel())
        
        # Compute perplexity over product space
        counts = torch.bincount(all_codes_cat, minlength=codebook_size)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        global_perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
    elif is_lmb:
        # Standard LMB (per-channel, not perchannel_fair)
        all_indices_cat = torch.cat([idx.view(-1) for idx in all_indices])
        global_active_codes = int(all_indices_cat.unique().numel())
        counts = torch.bincount(all_indices_cat.long(), minlength=1)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        global_perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
    else:
        # FSQ, VQ, LFQ, SimVQ: flat indices per token
        all_indices_cat = torch.cat([idx.view(-1) for idx in all_indices])
        global_active_codes = int(all_indices_cat.unique().numel())
        counts = torch.bincount(all_indices_cat.long(), minlength=1)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        global_perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
        # Utilization denominator: use index range (max_index+1) so we never under-report.
        # VQ/FSQ indices are 0..codebook_size-1; denominator must not exceed that.
        max_index = int(all_indices_cat.max().item())
        size_for_util = (max_index + 1) if (codebook_size is None or codebook_size <= 0) else min(codebook_size, max_index + 1)
        codebook_size = size_for_util  # use index-range denominator so utilization is not under-reported

    # Compute codebook utilization (shared for all branches)
    if codebook_size is not None and codebook_size > 0:
        codebook_util = 100.0 * global_active_codes / codebook_size
    else:
        codebook_util = 0.0

    # Empirical entropy (marginal) in bits: H = log2(perplexity)
    # Bits-per-token under i.i.d. marginal: same as marginal entropy (one code per token)
    entropy_bits = math.log2(global_perplexity) if global_perplexity > 0 else 0.0
    bits_per_token = entropy_bits

    # Bits per pixel: rate in bpp for principled rate–distortion (backbone 16× downsample)
    DOWNSAMPLE_FACTOR = 16
    tokens_per_pixel = 1.0 / (DOWNSAMPLE_FACTOR ** 2)
    bpp = bits_per_token * tokens_per_pixel if bits_per_token > 0 else 0.0

    metrics = {
        "rec_loss": total_rec_loss / max(num_batches, 1),
        "total_loss": total_total_loss / max(num_batches, 1),
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
        "lpips": float("nan") if (lpips_failed or total_lpips_samples == 0 or lpips_metric is None) else total_lpips / total_lpips_samples,
        "active_codes": global_active_codes,
        "codebook_util": codebook_util,
        "perplexity": global_perplexity,
        "entropy_bits": entropy_bits,
        "bits_per_token": bits_per_token,
        "bpp": bpp,
        "num_samples": samples_seen,
    }
    
    if aux_loss_name != "none":
        metrics[aux_loss_name] = total_aux_loss / max(num_batches, 1)
    
    # Compute rFID (reconstruction FID)
    if compute_fid:
        if FIDCalculator is None:
            print("Warning: FIDCalculator not available (torchvision required), skipping rFID")
            metrics["rfid"] = float("nan")
        else:
            try:
                if fid_calculator is None:
                    fid_calculator = FIDCalculator(device=device)
                
                rfid = fid_calculator.compute_rfid(
                    model=model,
                    dataloader=val_loader,
                    max_samples=fid_samples or max_samples,
                    show_progress=True,
                )
                metrics["rfid"] = rfid
            except Exception as e:
                print(f"Warning: Failed to compute rFID: {e}")
                metrics["rfid"] = float("nan")
    else:
        metrics["rfid"] = float("nan")
    
    # Save sample images
    if save_samples and output_dir and saved_originals:
        output_dir.mkdir(parents=True, exist_ok=True)
        
        originals = torch.cat(saved_originals, dim=0)[:num_save]
        recons = torch.cat(saved_recons, dim=0)[:num_save]
        
        # Denormalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        originals = originals * std + mean
        recons = recons * std + mean
        originals = originals.clamp(0, 1)
        recons = recons.clamp(0, 1)
        
        # Save grid
        if TORCHVISION_AVAILABLE:
            save_image(originals, output_dir / "originals.png", nrow=4)
            save_image(recons, output_dir / "reconstructions.png", nrow=4)
            # Save side-by-side comparison
            comparison = torch.cat([originals, recons], dim=0)
            save_image(comparison, output_dir / "comparison.png", nrow=4)
        else:
            # Fallback: save using PIL (simplified, just save first image)
            from PIL import Image as PILImage
            for i, (orig, recon) in enumerate(zip(originals[:4], recons[:4])):
                orig_pil = PILImage.fromarray((orig.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                recon_pil = PILImage.fromarray((recon.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
                orig_pil.save(output_dir / f"original_{i}.png")
                recon_pil.save(output_dir / f"reconstruction_{i}.png")
        
        print(f"\nSaved sample images to {output_dir}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained quantization model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--data-root", type=str, default="~/data/imagenet",
                        help="Path to ImageNet dataset")
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "test", "train"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to evaluate (default: all)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Number of data loading workers")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use")
    parser.add_argument("--save-samples", action="store_true",
                        help="Save sample reconstructions")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for samples (default: checkpoint dir)")
    
    # FID arguments
    parser.add_argument("--no-fid", action="store_true",
                        help="Skip FID computation (faster evaluation)")
    parser.add_argument("--fid-samples", type=int, default=10000,
                        help="Number of samples for FID computation")
    parser.add_argument("--write-eval-csv", type=str, default=None,
                        help="Append one row to this run dir's eval_metrics.csv (for comparison table)")
    
    args = parser.parse_args()
    
    # Set device
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, model_config, train_args, checkpoint = load_model_from_checkpoint(
        args.checkpoint, device
    )
    print(f"Model type: {checkpoint.get('model_type', train_args.model)}")
    print(f"Trained for {checkpoint.get('epoch', '?')} epochs")
    
    # Data transforms (use image_size from checkpoint config, default 224)
    image_size = int(getattr(train_args, "image_size", 224))
    if TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        # Fallback to PIL-based transforms
        def transform(img):
            img = img.resize((256, 256), Image.LANCZOS)
            # Center crop to image_size x image_size
            left = (256 - image_size) // 2
            top = (256 - image_size) // 2
            img = img.crop((left, top, left + image_size, top + image_size))
            # Convert to tensor and normalize
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            return img
    
    # Load dataset
    data_root = os.path.expanduser(args.data_root)
    data_path = os.path.join(data_root, args.split)
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        return
    
    # Try ImageFolder first, fall back to FlatImageDataset
    if TORCHVISION_AVAILABLE:
        try:
            dataset = datasets.ImageFolder(root=data_path, transform=transform)
            print(f"Loaded dataset using ImageFolder format")
        except (FileNotFoundError, RuntimeError):
            dataset = FlatImageDataset(root=data_path, transform=transform)
            print(f"Loaded dataset using flat directory format")
    else:
        dataset = FlatImageDataset(root=data_path, transform=transform)
        print(f"Loaded dataset using flat directory format (PIL)")
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Loaded {len(dataset)} images from {args.split} split")
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent.parent / "samples"
    
    # Get codebook size from model (different attributes for different quantizers)
    codebook_size = getattr(model, 'codebook_size', None)
    if codebook_size is None:
        codebook_size = getattr(model, 'num_codes', None)  # FSQ uses num_codes
    
    # Check model.quantize (VQ, SimVQ, LMB use this)
    if codebook_size is None and hasattr(model, 'quantize'):
        codebook_size = getattr(model.quantize, 'codebook_size', None)
        if codebook_size is None:
            codebook_size = getattr(model.quantize, 'num_codes', None)
        # LMB stores num_bins as .K
        if codebook_size is None:
            num_bins = getattr(model.quantize, 'K', None)
            if num_bins is not None:
                codebook_size = num_bins  # LMB: use num_bins as codebook size
    
    # Also check model.quantizer as fallback
    if codebook_size is None and hasattr(model, 'quantizer'):
        codebook_size = getattr(model.quantizer, 'codebook_size', None)
        if codebook_size is None:
            codebook_size = getattr(model.quantizer, 'num_codes', None)
    
    # Run evaluation
    print(f"\nEvaluating...")
    print(f"Metrics: rec_loss, PSNR↑, SSIM↑, LPIPS↓, rFID↓, codebook_util↑")
    compute_fid = not args.no_fid
    if compute_fid:
        print(f"Will compute rFID using up to {args.fid_samples} samples")
    
    metrics = evaluate(
        model,
        model_config,
        loader,
        device,
        max_samples=args.num_samples,
        save_samples=args.save_samples,
        output_dir=output_dir if args.save_samples else None,
        compute_fid=compute_fid,
        fid_samples=args.fid_samples,
        codebook_size=codebook_size,
    )
    
    # Print results
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 50)
    
    # Save results
    results_path = output_dir / "eval_results.json" if args.save_samples else None
    if results_path:
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"\nResults saved to {results_path}")
    
    # Append one row to run dir's eval_metrics.csv (for model_comparison_all_epochs)
    if args.write_eval_csv:
        run_dir = Path(args.write_eval_csv)
        run_dir.mkdir(parents=True, exist_ok=True)
        csv_path = run_dir / "eval_metrics.csv"
        epoch = checkpoint.get("epoch", 1)
        global_step = int(checkpoint.get("global_step", 0))
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "val_active_codes": metrics.get("active_codes"),
            "val_codebook_util": metrics.get("codebook_util"),
            "val_lpips": metrics.get("lpips"),
            "val_perplexity": metrics.get("perplexity"),
            "val_psnr": metrics.get("psnr"),
            "val_rec_loss": metrics.get("rec_loss"),
            "val_rfid": metrics.get("rfid"),
            "val_ssim": metrics.get("ssim"),
            "val_total_loss": metrics.get("total_loss"),
            "val_entropy_bits": metrics.get("entropy_bits"),
            "val_bits_per_token": metrics.get("bits_per_token"),
            "val_bpp": metrics.get("bpp"),
        }
        import csv as csv_module
        file_exists = csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            writer = csv_module.DictWriter(
                f,
                fieldnames=["epoch", "global_step", "val_active_codes", "val_codebook_util", "val_lpips",
                            "val_perplexity", "val_psnr", "val_rec_loss", "val_rfid", "val_ssim", "val_total_loss",
                            "val_entropy_bits", "val_bits_per_token", "val_bpp"],
                extrasaction="ignore",
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        print(f"Appended eval row to {csv_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate FSQ and SimVQ at matched effective capacity (active code count) from LMB.

1. Run reference (LMB) eval once → get K = active_codes.
2. FSQ: prune to K codes (remap indices outside top-K to nearest in top-K by code space).
3. SimVQ: mask to K codes (restrict argmin to top-K most-used codes).
4. Compare rFID / LPIPS at matched K.

Usage:
    python scripts/evaluate_matched_capacity.py \\
        --reference-checkpoint results/lmb/checkpoints/best_model.pt \\
        --fsq-checkpoint results/fsq/checkpoints/best_model.pt \\
        --sim-vq-checkpoint results/sim_vq/checkpoints/best_model.pt \\
        --data-root ~/data/imagenet --num-samples 5000 --fid-samples 5000

    # Optional: fix K manually (skip reference run)
    python scripts/evaluate_matched_capacity.py --capacity-k 5000 \\
        --fsq-checkpoint ... --sim-vq-checkpoint ...
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

TORCHVISION_AVAILABLE = False
try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

import numpy as np
from PIL import Image

from configs import MODEL_CONFIGS
from configs.base import LossOutput
from quantization.model import ModelConfig, UnifiedAutoEncoder, QuantizerConfig
from scripts.evaluation.evaluate import (
    load_model_from_checkpoint,
    evaluate,
    FlatImageDataset,
)
from scripts.evaluation.metrics import compute_psnr, compute_ssim, LPIPSMetric

FIDCalculator = None
try:
    from scripts.evaluation.fid import FIDCalculator
except ImportError:
    pass


def get_dataloader(
    data_root: str,
    split: str = "val",
    batch_size: int = 32,
    num_workers: int = 8,
    num_samples: Optional[int] = None,
):
    """Build validation dataloader (same transforms as evaluate.py)."""
    if TORCHVISION_AVAILABLE:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        def transform(img):
            img = img.resize((256, 256), Image.LANCZOS)
            left = (256 - 224) // 2
            top = (256 - 224) // 2
            img = img.crop((left, top, left + 224, top + 224))
            img = np.array(img).astype(np.float32) / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            return img

    data_path = os.path.join(os.path.expanduser(data_root), split)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    if TORCHVISION_AVAILABLE:
        try:
            dataset = datasets.ImageFolder(root=data_path, transform=transform)
        except (FileNotFoundError, RuntimeError):
            dataset = FlatImageDataset(root=data_path, transform=transform)
    else:
        dataset = FlatImageDataset(root=data_path, transform=transform)

    if num_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(num_samples, len(dataset))))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_K_from_reference(
    reference_checkpoint: str,
    val_loader: DataLoader,
    device: str,
    fid_samples: Optional[int] = None,
    fid_calculator=None,
) -> int:
    """Run reference (LMB) evaluation and return active_codes K (and LMB metrics including rFID)."""
    model, model_config, _args, checkpoint = load_model_from_checkpoint(
        reference_checkpoint, device
    )
    model_type = checkpoint.get("model_type", checkpoint.get("config", {}).get("model"))
    if model_type != "lmb":
        raise ValueError(
            f"Reference must be LMB to get active_codes K (got {model_type}). "
            "Use --capacity-k K to set K manually."
        )
    codebook_size = getattr(model.quantize, "codebook_size", None)
    if codebook_size is None and hasattr(model.quantize, "levels"):
        codebook_size = int(torch.tensor(model.quantize.levels).prod().item())
    if codebook_size is None:
        codebook_size = getattr(model.quantize, "K", None)

    metrics = evaluate(
        model,
        model_config,
        val_loader,
        device,
        max_samples=fid_samples,
        compute_fid=fid_calculator is not None,
        fid_calculator=fid_calculator,
        fid_samples=fid_samples,
        codebook_size=codebook_size,
    )
    K = int(metrics["active_codes"])
    print(f"Reference (LMB) active_codes K = {K}")
    # Return LMB metrics for comparison (same K)
    lmb_metrics = {
        "rec_loss": metrics.get("rec_loss", float("nan")),
        "psnr": metrics.get("psnr", float("nan")),
        "ssim": metrics.get("ssim", float("nan")),
        "lpips": metrics.get("lpips", float("nan")),
        "active_codes": K,
        "num_samples": metrics.get("num_samples", 0),
        "rfid": metrics.get("rfid", float("nan")),
        "note": "reference (natural K)",
    }
    return K, lmb_metrics


# ---------------------------------------------------------------------------
# FSQ: prune to top-K indices (remap out-of-set to nearest in code space)
# ---------------------------------------------------------------------------


def _collect_fsq_index_counts(model, loader, device, max_samples: Optional[int] = None):
    """One pass: collect product-index counts for FSQ."""
    model.eval()
    counts = {}
    n = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="FSQ index usage", leave=False):
            x = x.to(device)
            _recon, indices = model(x)
            indices = indices.view(-1).long().cpu()
            for idx in indices.tolist():
                counts[idx] = counts.get(idx, 0) + 1
            n += x.size(0)
            if max_samples and n >= max_samples:
                break
    return counts


def _fsq_remap_indices_to_top_k(
    model,
    indices: torch.Tensor,
    top_k_indices: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Remap indices not in top_k to nearest in top_k (in code space)."""
    # top_k_indices: (K,) on device
    # indices: (B, H, W) or (N,) on device
    flat = indices.view(-1).long()
    in_top_k = (flat.unsqueeze(1) == top_k_indices.unsqueeze(0)).any(1)
    if in_top_k.all():
        return indices

    # Codes for top-K: (K, dim) in decoder space
    q = model.quantize
    top_k_codes = q.indices_to_codes(top_k_indices)
    if top_k_codes.dim() > 2:
        top_k_codes = top_k_codes.reshape(top_k_codes.size(0), -1)
    # Out-of-set indices -> their codes
    out_mask = ~in_top_k
    out_indices = flat[out_mask]
    out_codes = q.indices_to_codes(out_indices)
    if out_codes.dim() > 2:
        out_codes = out_codes.reshape(out_codes.size(0), -1)
    # out_codes (M, C), top_k_codes (K, C) -> nearest in top_k
    dist = torch.cdist(out_codes.float(), top_k_codes.float())
    nearest = dist.argmin(dim=1)
    remapped_flat = flat.clone()
    remapped_flat[out_mask] = top_k_indices[nearest]
    return remapped_flat.view_as(indices)


class FSQCapacityLimitWrapper(torch.nn.Module):
    """Wrapper that runs FSQ forward then remaps indices to top-K and decodes."""

    def __init__(self, model: UnifiedAutoEncoder, top_k_indices: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("_top_k", top_k_indices)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.model.enc(x)
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_norm = self.model.pre_quant(z_norm)
            _, indices = self.model.quantize(z_norm)
            remapped = _fsq_remap_indices_to_top_k(
                self.model, indices, self._top_k, x.device
            )
            z_q = self.model.quantize.indices_to_codes(remapped)
            z_q = self.model.post_quant(z_q)
            recon = self.model.dec(z_q)
        return recon, remapped


def evaluate_fsq_at_capacity_K(
    model: UnifiedAutoEncoder,
    model_config,
    val_loader: DataLoader,
    device: str,
    K: int,
    max_samples: Optional[int] = None,
    fid_samples: Optional[int] = None,
    fid_calculator=None,
    lpips_metric=None,
) -> dict:
    """Run FSQ eval with effective capacity limited to K (prune to top-K codes)."""
    counts = _collect_fsq_index_counts(model, val_loader, device, max_samples=fid_samples or max_samples)
    if len(counts) < K:
        top_k_indices = torch.tensor(sorted(counts.keys()), device=device, dtype=torch.long)
        K_actual = len(top_k_indices)
        print(f"  FSQ: only {len(counts)} unique indices, using K={K_actual}")
    else:
        sorted_indices = sorted(counts.keys(), key=lambda i: -counts[i])
        top_k_indices = torch.tensor(sorted_indices[:K], device=device, dtype=torch.long)
        K_actual = K

    wrapper = FSQCapacityLimitWrapper(model, top_k_indices)
    wrapper.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_lpips_samples = 0
    lpips_failed = False
    total_rec_loss = 0.0
    num_batches = 0
    n_seen = 0

    with torch.no_grad():
        for x, _ in tqdm(val_loader, desc="FSQ@K eval", leave=False):
            if max_samples and n_seen >= max_samples:
                break
            x = x.to(device)
            recon, _ = wrapper(x)
            rec_loss = (recon - x).float().abs().mean().item()
            total_rec_loss += rec_loss
            x_denorm = (x * std + mean).clamp(0, 1)
            recon_denorm = (recon * std + mean).clamp(0, 1)
            total_psnr += compute_psnr(x_denorm, recon_denorm)
            total_ssim += compute_ssim(x_denorm, recon_denorm)
            if lpips_metric is not None and not lpips_failed:
                try:
                    batch_lpips = lpips_metric.compute(x_denorm, recon_denorm)
                    total_lpips += batch_lpips * x.size(0)
                    total_lpips_samples += x.size(0)
                except Exception as e:
                    if num_batches == 1:
                        print(f"  Warning: LPIPS failed: {e}")
                    lpips_failed = True
            num_batches += 1
            n_seen += x.size(0)

    metrics = {
        "rec_loss": total_rec_loss / max(num_batches, 1),
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
        "lpips": float("nan") if (lpips_failed or total_lpips_samples == 0 or lpips_metric is None) else total_lpips / total_lpips_samples,
        "active_codes": K_actual,
        "num_samples": n_seen,
    }

    if fid_calculator is not None and FIDCalculator is not None:
        try:
            metrics["rfid"] = fid_calculator.compute_rfid(
                wrapper,
                val_loader,
                max_samples=fid_samples or max_samples,
                show_progress=True,
            )
        except Exception as e:
            print(f"  Warning: rFID failed: {e}")
            metrics["rfid"] = float("nan")
    else:
        metrics["rfid"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# SimVQ: mask to top-K codes (argmin only over top-K)
# ---------------------------------------------------------------------------


def _collect_simvq_index_counts(model, loader, device, max_samples: Optional[int] = None):
    """One pass: collect code indices for SimVQ."""
    model.eval()
    counts = {}
    n = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="SimVQ index usage", leave=False):
            x = x.to(device)
            _recon, indices, _ = model(x)
            indices = indices.view(-1).long().cpu()
            for idx in indices.tolist():
                counts[idx] = counts.get(idx, 0) + 1
            n += x.size(0)
            if max_samples and n >= max_samples:
                break
    return counts


class SimVQCapacityLimitWrapper(torch.nn.Module):
    """Wrapper that restricts SimVQ to top-K codes (argmin over K only)."""

    def __init__(self, model: UnifiedAutoEncoder, top_k_indices: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("_top_k", top_k_indices)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.model.enc(x)
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            q = self.model.quantize
            # z_norm: (B, C, H, W) -> (B*H*W, C) for cdist
            z_norm_flat = z_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)
            codebook = q.codebook
            top_k = self._top_k
            dist_full = torch.cdist(z_norm_flat.float(), codebook.float())
            dist_k = dist_full[:, top_k]
            idx_k = dist_k.argmin(dim=1)
            indices = top_k[idx_k].view(B, H, W)
            quantized = q.indices_to_codes(indices)
            recon = self.model.dec(quantized)
        return recon, indices


def evaluate_simvq_at_capacity_K(
    model: UnifiedAutoEncoder,
    model_config,
    val_loader: DataLoader,
    device: str,
    K: int,
    max_samples: Optional[int] = None,
    fid_samples: Optional[int] = None,
    fid_calculator=None,
    lpips_metric=None,
) -> dict:
    """Run SimVQ eval with effective capacity limited to K (mask to top-K codes)."""
    counts = _collect_simvq_index_counts(model, val_loader, device, max_samples=fid_samples or max_samples)
    if len(counts) < K:
        top_k_indices = torch.tensor(sorted(counts.keys()), device=device, dtype=torch.long)
        K_actual = len(top_k_indices)
        print(f"  SimVQ: only {len(counts)} unique indices, using K={K_actual}")
    else:
        sorted_indices = sorted(counts.keys(), key=lambda i: -counts[i])
        top_k_indices = torch.tensor(sorted_indices[:K], device=device, dtype=torch.long)
        K_actual = K

    wrapper = SimVQCapacityLimitWrapper(model, top_k_indices)
    wrapper.eval()

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0
    total_lpips_samples = 0
    lpips_failed = False
    total_rec_loss = 0.0
    num_batches = 0
    n_seen = 0

    with torch.no_grad():
        for x, _ in tqdm(val_loader, desc="SimVQ@K eval", leave=False):
            if max_samples and n_seen >= max_samples:
                break
            x = x.to(device)
            recon, _ = wrapper(x)
            rec_loss = (recon - x).float().abs().mean().item()
            total_rec_loss += rec_loss
            x_denorm = (x * std + mean).clamp(0, 1)
            recon_denorm = (recon * std + mean).clamp(0, 1)
            total_psnr += compute_psnr(x_denorm, recon_denorm)
            total_ssim += compute_ssim(x_denorm, recon_denorm)
            if lpips_metric is not None and not lpips_failed:
                try:
                    batch_lpips = lpips_metric.compute(x_denorm, recon_denorm)
                    total_lpips += batch_lpips * x.size(0)
                    total_lpips_samples += x.size(0)
                except Exception as e:
                    if num_batches == 1:
                        print(f"  Warning: LPIPS failed: {e}")
                    lpips_failed = True
            num_batches += 1
            n_seen += x.size(0)

    metrics = {
        "rec_loss": total_rec_loss / max(num_batches, 1),
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
        "lpips": float("nan") if (lpips_failed or total_lpips_samples == 0 or lpips_metric is None) else total_lpips / total_lpips_samples,
        "active_codes": K_actual,
        "num_samples": n_seen,
    }

    if fid_calculator is not None and FIDCalculator is not None:
        try:
            metrics["rfid"] = fid_calculator.compute_rfid(
                wrapper,
                val_loader,
                max_samples=fid_samples or max_samples,
                show_progress=True,
            )
        except Exception as e:
            print(f"  Warning: rFID failed: {e}")
            metrics["rfid"] = float("nan")
    else:
        metrics["rfid"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# LFQ: prune to top-K indices (remap out-of-set to nearest in code space, like FSQ)
# ---------------------------------------------------------------------------


def _collect_lfq_index_counts(model, loader, device, max_samples: Optional[int] = None):
    """One pass: collect index counts for LFQ. LFQ returns (B, C, H, W); linear index = sum over C."""
    model.eval()
    counts = {}
    n = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="LFQ index usage", leave=False):
            x = x.to(device)
            _recon, indices, _ = model(x)
            # indices (B, C, H, W) -> linear index (B, H, W) = sum of bit channels
            linear = indices.sum(dim=1).view(-1).long().cpu()
            for idx in linear.tolist():
                counts[idx] = counts.get(idx, 0) + 1
            n += x.size(0)
            if max_samples and n >= max_samples:
                break
    return counts


def _lfq_remap_indices_to_top_k(model, indices_linear: torch.Tensor, top_k_indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Remap linear indices not in top_k to nearest in top_k (in code space). indices_linear: (B, H, W)."""
    flat = indices_linear.view(-1).long()
    in_top_k = (flat.unsqueeze(1) == top_k_indices.unsqueeze(0)).any(1)
    if in_top_k.all():
        return indices_linear
    q = model.quantize
    top_k_codes = q.indices_to_codes(top_k_indices, project_out=True)
    if top_k_codes.dim() > 2:
        top_k_codes = top_k_codes.reshape(top_k_codes.size(0), -1)
    out_mask = ~in_top_k
    out_indices = flat[out_mask]
    out_codes = q.indices_to_codes(out_indices, project_out=True)
    if out_codes.dim() > 2:
        out_codes = out_codes.reshape(out_codes.size(0), -1)
    dist = torch.cdist(out_codes.float(), top_k_codes.float())
    nearest = dist.argmin(dim=1)
    remapped_flat = flat.clone()
    remapped_flat[out_mask] = top_k_indices[nearest]
    return remapped_flat.view_as(indices_linear)


class LFQCapacityLimitWrapper(torch.nn.Module):
    """Wrapper that runs LFQ forward then remaps indices to top-K and decodes."""

    def __init__(self, model: UnifiedAutoEncoder, top_k_indices: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("_top_k", top_k_indices)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.model.enc(x)
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_norm = self.model.pre_quant(z_norm)
            _, indices, _ = self.model.quantize(z_norm)
            # indices (B, C, H, W) -> linear (B, H, W)
            linear = indices.sum(dim=1)
            remapped = _lfq_remap_indices_to_top_k(self.model, linear, self._top_k, x.device)
            z_q = self.model.quantize.indices_to_codes(remapped, project_out=True)
            # Ensure (B, C, H, W): LFQ may return channel-last or quantizer dim may differ from post_quant
            expected_C = self.model.post_quant.weight.shape[1]
            if z_q.dim() == 4 and z_q.shape[1] != expected_C:
                # Channel-last (B, H, W, C): only if last dim is C and middle dims look spatial (H==W)
                if z_q.shape[-1] == expected_C and z_q.shape[1] == z_q.shape[2]:
                    z_q = z_q.permute(0, 3, 1, 2)
                elif getattr(self.model.quantize, "has_projections", False) and z_q.shape[1] == self.model.quantize.dim:
                    # Quantizer returns dim channels; post_quant expects codebook_dims; use project_in
                    z_q = self.model.quantize.project_in(
                        z_q.permute(0, 2, 3, 1).reshape(-1, z_q.shape[1])
                    ).view(B, H, W, -1).permute(0, 3, 1, 2)
                elif z_q.shape[1] > expected_C:
                    # Checkpoint quantizer output dim > built post_quant (e.g. config mismatch): truncate
                    z_q = z_q[:, :expected_C, :, :]
            z_q = self.model.post_quant(z_q)
            recon = self.model.dec(z_q)
        return recon, remapped


def evaluate_lfq_at_capacity_K(
    model: UnifiedAutoEncoder,
    model_config,
    val_loader: DataLoader,
    device: str,
    K: int,
    max_samples: Optional[int] = None,
    fid_samples: Optional[int] = None,
    fid_calculator=None,
    lpips_metric=None,
) -> dict:
    """Run LFQ eval with effective capacity limited to K (prune to top-K codes)."""
    counts = _collect_lfq_index_counts(model, val_loader, device, max_samples=fid_samples or max_samples)
    if len(counts) < K:
        top_k_indices = torch.tensor(sorted(counts.keys()), device=device, dtype=torch.long)
        K_actual = len(top_k_indices)
        print(f"  LFQ: only {len(counts)} unique indices, using K={K_actual}")
    else:
        sorted_indices = sorted(counts.keys(), key=lambda i: -counts[i])
        top_k_indices = torch.tensor(sorted_indices[:K], device=device, dtype=torch.long)
        K_actual = K
    wrapper = LFQCapacityLimitWrapper(model, top_k_indices)
    wrapper.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    total_psnr = total_ssim = total_lpips = total_rec_loss = 0.0
    total_lpips_samples = 0
    lpips_failed = False
    num_batches = n_seen = 0
    with torch.no_grad():
        for x, _ in tqdm(val_loader, desc="LFQ@K eval", leave=False):
            if max_samples and n_seen >= max_samples:
                break
            x = x.to(device)
            recon, _ = wrapper(x)
            total_rec_loss += (recon - x).float().abs().mean().item()
            x_denorm = (x * std + mean).clamp(0, 1)
            recon_denorm = (recon * std + mean).clamp(0, 1)
            total_psnr += compute_psnr(x_denorm, recon_denorm)
            total_ssim += compute_ssim(x_denorm, recon_denorm)
            if lpips_metric and not lpips_failed:
                try:
                    batch_lpips = lpips_metric.compute(x_denorm, recon_denorm)
                    total_lpips += batch_lpips * x.size(0)
                    total_lpips_samples += x.size(0)
                except Exception as e:
                    if num_batches == 1:
                        print(f"  Warning: LPIPS failed: {e}")
                    lpips_failed = True
            num_batches += 1
            n_seen += x.size(0)
    metrics = {
        "rec_loss": total_rec_loss / max(num_batches, 1),
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
        "lpips": float("nan") if (lpips_failed or total_lpips_samples == 0 or not lpips_metric) else total_lpips / total_lpips_samples,
        "active_codes": K_actual,
        "num_samples": n_seen,
        "note": "at matched capacity K (pruned)",
    }
    if fid_calculator and FIDCalculator:
        try:
            metrics["rfid"] = fid_calculator.compute_rfid(wrapper, val_loader, max_samples=fid_samples or max_samples, show_progress=True)
        except Exception as e:
            print(f"  Warning: rFID failed: {e}")
            metrics["rfid"] = float("nan")
    else:
        metrics["rfid"] = float("nan")
    return metrics


# ---------------------------------------------------------------------------
# VQ: mask to top-K codes (same as SimVQ; use get_output_from_indices)
# ---------------------------------------------------------------------------


def _collect_vq_index_counts(model, loader, device, max_samples: Optional[int] = None):
    """One pass: collect code indices for VQ."""
    model.eval()
    counts = {}
    n = 0
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="VQ index usage", leave=False):
            x = x.to(device)
            _recon, indices, _ = model(x)
            indices = indices.view(-1).long().cpu()
            for idx in indices.tolist():
                counts[idx] = counts.get(idx, 0) + 1
            n += x.size(0)
            if max_samples and n >= max_samples:
                break
    return counts


class VQCapacityLimitWrapper(torch.nn.Module):
    """Wrapper that restricts VQ to top-K codes (argmin over K only)."""

    def __init__(self, model: UnifiedAutoEncoder, top_k_indices: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("_top_k", top_k_indices)

    def forward(self, x: torch.Tensor):
        with torch.no_grad():
            z = self.model.enc(x)
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            q = self.model.quantize
            z_norm_flat = z_norm.permute(0, 2, 3, 1).reshape(B * H * W, C)
            codebook = q.codebook
            top_k = self._top_k
            dist_full = torch.cdist(z_norm_flat.float(), codebook.float())
            dist_k = dist_full[:, top_k]
            idx_k = dist_k.argmin(dim=1)
            indices = top_k[idx_k].view(B, H, W)
            quantized = q.get_output_from_indices(indices)
            recon = self.model.dec(quantized)
        return recon, indices


def evaluate_vq_at_capacity_K(
    model: UnifiedAutoEncoder,
    model_config,
    val_loader: DataLoader,
    device: str,
    K: int,
    max_samples: Optional[int] = None,
    fid_samples: Optional[int] = None,
    fid_calculator=None,
    lpips_metric=None,
) -> dict:
    """Run VQ eval with effective capacity limited to K (mask to top-K codes)."""
    counts = _collect_vq_index_counts(model, val_loader, device, max_samples=fid_samples or max_samples)
    if len(counts) < K:
        top_k_indices = torch.tensor(sorted(counts.keys()), device=device, dtype=torch.long)
        K_actual = len(top_k_indices)
        print(f"  VQ: only {len(counts)} unique indices, using K={K_actual}")
    else:
        sorted_indices = sorted(counts.keys(), key=lambda i: -counts[i])
        top_k_indices = torch.tensor(sorted_indices[:K], device=device, dtype=torch.long)
        K_actual = K
    wrapper = VQCapacityLimitWrapper(model, top_k_indices)
    wrapper.eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    total_psnr = total_ssim = total_lpips = total_rec_loss = 0.0
    total_lpips_samples = 0
    lpips_failed = False
    num_batches = n_seen = 0
    with torch.no_grad():
        for x, _ in tqdm(val_loader, desc="VQ@K eval", leave=False):
            if max_samples and n_seen >= max_samples:
                break
            x = x.to(device)
            recon, _ = wrapper(x)
            total_rec_loss += (recon - x).float().abs().mean().item()
            x_denorm = (x * std + mean).clamp(0, 1)
            recon_denorm = (recon * std + mean).clamp(0, 1)
            total_psnr += compute_psnr(x_denorm, recon_denorm)
            total_ssim += compute_ssim(x_denorm, recon_denorm)
            if lpips_metric and not lpips_failed:
                try:
                    batch_lpips = lpips_metric.compute(x_denorm, recon_denorm)
                    total_lpips += batch_lpips * x.size(0)
                    total_lpips_samples += x.size(0)
                except Exception as e:
                    if num_batches == 1:
                        print(f"  Warning: LPIPS failed: {e}")
                    lpips_failed = True
            num_batches += 1
            n_seen += x.size(0)
    metrics = {
        "rec_loss": total_rec_loss / max(num_batches, 1),
        "psnr": total_psnr / max(num_batches, 1),
        "ssim": total_ssim / max(num_batches, 1),
        "lpips": float("nan") if (lpips_failed or total_lpips_samples == 0 or not lpips_metric) else total_lpips / total_lpips_samples,
        "active_codes": K_actual,
        "num_samples": n_seen,
        "note": "at matched capacity K (masked)",
    }
    if fid_calculator and FIDCalculator:
        try:
            metrics["rfid"] = fid_calculator.compute_rfid(wrapper, val_loader, max_samples=fid_samples or max_samples, show_progress=True)
        except Exception as e:
            print(f"  Warning: rFID failed: {e}")
            metrics["rfid"] = float("nan")
    else:
        metrics["rfid"] = float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate at matched effective capacity (LMB K → all at K); compare to LMB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--reference-checkpoint", type=str, default=None,
                        help="LMB checkpoint to get active_codes K from (also stores LMB metrics)")
    parser.add_argument("--capacity-k", type=int, default=None,
                        help="Use this K directly (skip reference run)")
    parser.add_argument("--lmb-checkpoint", type=str, default=None,
                        help="LMB checkpoint to eval for comparison (only when using --capacity-k)")
    parser.add_argument("--fsq-checkpoint", type=str, default=None,
                        help="FSQ checkpoint to evaluate at capacity K")
    parser.add_argument("--sim-vq-checkpoint", type=str, default=None,
                        help="SimVQ checkpoint to evaluate at capacity K")
    parser.add_argument("--lfq-checkpoint", type=str, default=None,
                        help="LFQ checkpoint to evaluate at capacity K")
    parser.add_argument("--vq-checkpoint", type=str, default=None,
                        help="VQ checkpoint to evaluate at capacity K")
    parser.add_argument("--data-root", type=str, default="~/data/imagenet")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Max samples for metric loop")
    parser.add_argument("--fid-samples", type=int, default=10000,
                        help="Samples for rFID (and for usage count when getting K)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default=None,
                        help="Save comparison JSON here")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    fid_calculator = FIDCalculator(device=device) if FIDCalculator else None
    lpips_metric = None
    try:
        lpips_metric = LPIPSMetric(device=device)
    except Exception:
        pass

    lmb_metrics = None
    if args.capacity_k is not None:
        K = args.capacity_k
        print(f"Using fixed K = {K}")
    elif args.reference_checkpoint:
        loader = get_dataloader(
            args.data_root,
            args.split,
            args.batch_size,
            args.num_workers,
            num_samples=args.fid_samples,
        )
        K, lmb_metrics = get_K_from_reference(
            args.reference_checkpoint,
            loader,
            device,
            fid_samples=args.fid_samples,
            fid_calculator=fid_calculator,
        )
    else:
        print("Provide --reference-checkpoint or --capacity-k")
        return

    results = {"K": K, "reference": args.reference_checkpoint or f"fixed_{K}"}
    if lmb_metrics is not None:
        results["lmb"] = lmb_metrics
        print(f"  LMB (reference): rFID={lmb_metrics.get('rfid', float('nan')):.2f}  PSNR={lmb_metrics.get('psnr', 0):.2f}  SSIM={lmb_metrics.get('ssim', 0):.4f}")

    if args.fsq_checkpoint:
        print(f"\nFSQ at K={K} ({args.fsq_checkpoint})")
        model, model_config, _, _ = load_model_from_checkpoint(args.fsq_checkpoint, device)
        loader = get_dataloader(
            args.data_root, args.split, args.batch_size, args.num_workers,
            num_samples=args.num_samples,
        )
        fsq_metrics = evaluate_fsq_at_capacity_K(
            model,
            model_config,
            loader,
            device,
            K,
            max_samples=args.num_samples,
            fid_samples=args.fid_samples,
            fid_calculator=fid_calculator,
            lpips_metric=lpips_metric,
        )
        results["fsq"] = fsq_metrics
        print(f"  rFID={fsq_metrics.get('rfid', float('nan')):.2f}  LPIPS={fsq_metrics.get('lpips', float('nan')):.4f}  PSNR={fsq_metrics.get('psnr', 0):.2f}  active={fsq_metrics.get('active_codes')}")

    if args.sim_vq_checkpoint:
        print(f"\nSimVQ at K={K} ({args.sim_vq_checkpoint})")
        model, model_config, _, _ = load_model_from_checkpoint(args.sim_vq_checkpoint, device)
        loader = get_dataloader(
            args.data_root, args.split, args.batch_size, args.num_workers,
            num_samples=args.num_samples,
        )
        simvq_metrics = evaluate_simvq_at_capacity_K(
            model,
            model_config,
            loader,
            device,
            K,
            max_samples=args.num_samples,
            fid_samples=args.fid_samples,
            fid_calculator=fid_calculator,
            lpips_metric=lpips_metric,
        )
        results["sim_vq"] = simvq_metrics
        print(f"  rFID={simvq_metrics.get('rfid', float('nan')):.2f}  LPIPS={simvq_metrics.get('lpips', float('nan')):.4f}  PSNR={simvq_metrics.get('psnr', 0):.2f}  active={simvq_metrics.get('active_codes')}")

    if args.lfq_checkpoint:
        print(f"\nLFQ at K={K} ({args.lfq_checkpoint})")
        model, model_config, _, _ = load_model_from_checkpoint(args.lfq_checkpoint, device)
        loader = get_dataloader(
            args.data_root, args.split, args.batch_size, args.num_workers,
            num_samples=args.num_samples,
        )
        lfq_metrics = evaluate_lfq_at_capacity_K(
            model, model_config, loader, device, K,
            max_samples=args.num_samples, fid_samples=args.fid_samples,
            fid_calculator=fid_calculator, lpips_metric=lpips_metric,
        )
        results["lfq"] = lfq_metrics
        print(f"  rFID={lfq_metrics.get('rfid', float('nan')):.2f}  LPIPS={lfq_metrics.get('lpips', float('nan')):.4f}  PSNR={lfq_metrics.get('psnr', 0):.2f}  active={lfq_metrics.get('active_codes')}")

    if args.vq_checkpoint:
        print(f"\nVQ at K={K} ({args.vq_checkpoint})")
        model, model_config, _, _ = load_model_from_checkpoint(args.vq_checkpoint, device)
        loader = get_dataloader(
            args.data_root, args.split, args.batch_size, args.num_workers,
            num_samples=args.num_samples,
        )
        vq_metrics = evaluate_vq_at_capacity_K(
            model, model_config, loader, device, K,
            max_samples=args.num_samples, fid_samples=args.fid_samples,
            fid_calculator=fid_calculator, lpips_metric=lpips_metric,
        )
        results["vq"] = vq_metrics
        print(f"  rFID={vq_metrics.get('rfid', float('nan')):.2f}  LPIPS={vq_metrics.get('lpips', float('nan')):.4f}  PSNR={vq_metrics.get('psnr', 0):.2f}  active={vq_metrics.get('active_codes')}")

    # If --capacity-k and --lmb-checkpoint: run LMB eval for comparison
    if args.capacity_k is not None and args.lmb_checkpoint and "lmb" not in results:
        print(f"\nLMB (for comparison at same K) ({args.lmb_checkpoint})")
        model, model_config, _, checkpoint = load_model_from_checkpoint(args.lmb_checkpoint, device)
        loader = get_dataloader(
            args.data_root, args.split, args.batch_size, args.num_workers,
            num_samples=args.num_samples,
        )
        codebook_size = getattr(model.quantize, "codebook_size", None) or getattr(model.quantize, "K", None)
        if hasattr(model.quantize, "levels"):
            codebook_size = int(torch.tensor(model.quantize.levels).prod().item())
        lmb_metrics = evaluate(model, model_config, loader, device, max_samples=args.num_samples, compute_fid=bool(fid_calculator), fid_samples=args.fid_samples, codebook_size=codebook_size)
        results["lmb"] = {
            "rec_loss": lmb_metrics.get("rec_loss", float("nan")),
            "psnr": lmb_metrics.get("psnr", float("nan")),
            "ssim": lmb_metrics.get("ssim", float("nan")),
            "lpips": lmb_metrics.get("lpips", float("nan")),
            "active_codes": int(lmb_metrics.get("active_codes", 0)),
            "num_samples": lmb_metrics.get("num_samples", 0),
            "rfid": lmb_metrics.get("rfid", float("nan")),
            "note": "reference (natural active codes)",
        }
        print(f"  rFID={results['lmb']['rfid']:.2f}  PSNR={results['lmb']['psnr']:.2f}  SSIM={results['lmb']['ssim']:.4f}  active={results['lmb']['active_codes']}")

    print("\n" + "=" * 60)
    print("Matched effective capacity (active codes = K) — comparison to LMB")
    print("=" * 60)
    print(f"  K = {K}")
    order = ["lmb", "fsq", "sim_vq", "lfq", "vq"]
    for name in order:
        if name not in results:
            continue
        m = results[name]
        print(f"  {name}: rFID={m.get('rfid', float('nan')):.2f}  LPIPS={m.get('lpips', float('nan')):.4f}  PSNR={m.get('psnr', 0):.2f}  SSIM={m.get('ssim', 0):.4f}  active_codes={m.get('active_codes')}")
    for name, data in results.items():
        if name in ("K", "reference") or name in order:
            continue
        m = data
        print(f"  {name}: rFID={m.get('rfid', float('nan')):.2f}  LPIPS={m.get('lpips', float('nan')):.4f}  PSNR={m.get('psnr', 0):.2f}  SSIM={m.get('ssim', 0):.4f}  active_codes={m.get('active_codes')}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

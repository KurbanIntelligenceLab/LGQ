#!/usr/bin/env python3
"""
Per-class / per-frequency robustness: Are fewer active codes hurting rare or fine details?

Analyzes reconstruction quality by:
1. Spatial frequency band (low=structure, high=texture)
2. Edge vs smooth regions

Plots:
- PSNR vs spatial frequency band (low, mid-low, mid-high, high)
- Edge-region PSNR vs flat-region PSNR

Usage:
    python scripts/plot_per_frequency_and_region.py --data-root data/imagenet
    python scripts/plot_per_frequency_and_region.py --num-samples 500 --output plots/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Default checkpoints (16K main runs)
DEFAULT_CHECKPOINTS = {
    "FSQ": "results/fsq/fsq_lv16-16-8-8_bs32_lr3e-4_dim128_20260123_184222_deab/checkpoints/best_model.pt",
    "SimVQ": "results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32/checkpoints/best_model.pt",
    "LMB": "results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd/checkpoints/best_model.pt",
    "LMB-Fixed": "results/lmb/lmb_fixed_init/checkpoints/best_model.pt",
    "VQ": "results/vq/vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0/checkpoints/best_model.pt",
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --- Dataset & transform ---
class FlatImageDataset(Dataset):
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}

    def __init__(self, root: str, transform=None, max_images=None):
        root_path = Path(root)
        self.image_paths = []
        for ext in self.EXTENSIONS:
            self.image_paths.extend(root_path.glob(f'*{ext}'))
            self.image_paths.extend(root_path.glob(f'*{ext.upper()}'))
            self.image_paths.extend(root_path.glob(f'**/*{ext}'))
            self.image_paths.extend(root_path.glob(f'**/*{ext.upper()}'))
        self.image_paths = sorted(set(self.image_paths))
        if max_images:
            self.image_paths = self.image_paths[:max_images]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_transform(image_size: int = 128):
    def _t(pil_img):
        img = pil_img.resize((image_size, image_size), Image.LANCZOS)
        x = np.array(img, dtype=np.float32) / 255.0
        x = (x - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.from_numpy(x).permute(2, 0, 1).float()
    return _t


# --- Frequency band decomposition ---
def get_radial_freq_mask(H: int, W: int, r_min: float, r_max: float, device: torch.device) -> torch.Tensor:
    """Create a radial mask in frequency space. r_min, r_max in [0, 1] as fraction of Nyquist."""
    cy, cx = H // 2, W // 2
    y = torch.arange(H, device=device, dtype=torch.float32) - cy
    x = torch.arange(W, device=device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    r = torch.sqrt(yy ** 2 + xx ** 2)
    nyquist = min(H, W) / 2
    r_norm = r / nyquist
    mask = ((r_norm >= r_min) & (r_norm < r_max)).float()
    return mask  # [H, W]


def bandpass_filter(img: torch.Tensor, r_min: float, r_max: float) -> torch.Tensor:
    """Apply bandpass filter: keep frequencies in [r_min, r_max] (fraction of Nyquist)."""
    # img: [B, C, H, W]
    B, C, H, W = img.shape
    device = img.device
    mask = get_radial_freq_mask(H, W, r_min, r_max, device)  # [H, W]
    # FFT per channel
    f = torch.fft.fft2(img, dim=(-2, -1))
    f = torch.fft.fftshift(f, dim=(-2, -1))
    mask_bc = mask.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    f_filtered = f * mask_bc
    f_filtered = torch.fft.ifftshift(f_filtered, dim=(-2, -1))
    out = torch.fft.ifft2(f_filtered, dim=(-2, -1)).real
    return out


# Frequency bands: low (structure), mid-low, mid-high, high (fine texture)
FREQ_BANDS = [
    ("Low\n(structure)", 0.0, 0.25),
    ("Mid-Low", 0.25, 0.5),
    ("Mid-High", 0.5, 0.75),
    ("High\n(texture)", 0.75, 1.0),
]


def compute_psnr_masked(orig: torch.Tensor, recon: torch.Tensor, mask: torch.Tensor, eps: float = 1e-10) -> float:
    """PSNR computed only over masked pixels. mask: [B,C,H,W] or [H,W], 1=include."""
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0).expand_as(orig)
    diff_sq = (orig - recon) ** 2
    num = (mask * diff_sq).sum()
    denom = mask.sum().clamp_min(eps)
    mse = (num / denom).item()
    if mse <= 0:
        return float('inf')
    return float(10 * np.log10(1.0 ** 2 / mse))


def compute_psnr_band(orig: torch.Tensor, recon: torch.Tensor, r_min: float, r_max: float) -> float:
    """PSNR in a specific frequency band."""
    o_band = bandpass_filter(orig, r_min, r_max)
    r_band = bandpass_filter(recon, r_min, r_max)
    return compute_psnr_masked(o_band, r_band, torch.ones_like(orig))  # full spatial, band-limited freq


# --- Edge vs smooth ---
def get_edge_smooth_masks(img: torch.Tensor, edge_percentile: float = 80) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Segment image into edge and smooth regions using gradient magnitude.
    img: [B, C, H, W] in [0, 1]
    Returns edge_mask, smooth_mask (each [B, 1, H, W], values 0 or 1).
    """
    gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    sobel_x = F.conv2d(gray, torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=gray.dtype, device=gray.device) / 4, padding=1)
    sobel_y = F.conv2d(gray, torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=gray.dtype, device=gray.device) / 4, padding=1)
    grad = torch.sqrt(sobel_x ** 2 + sobel_y ** 2 + 1e-8)
    B, _, H, W = grad.shape
    flat = grad.view(B, -1)
    thresh = torch.quantile(flat, edge_percentile / 100.0, dim=1, keepdim=True)
    thresh = thresh.view(B, 1, 1, 1)
    edge_mask = (grad >= thresh).float()
    smooth_mask = (grad < thresh).float()
    edge_mask = edge_mask.expand(-1, 3, -1, -1)
    smooth_mask = smooth_mask.expand(-1, 3, -1, -1)
    return edge_mask, smooth_mask


def compute_psnr_edge_smooth(orig: torch.Tensor, recon: torch.Tensor, edge_pct: float = 80) -> Tuple[float, float]:
    """PSNR in edge regions and smooth regions."""
    edge_mask, smooth_mask = get_edge_smooth_masks(orig, edge_percentile=edge_pct)
    psnr_edge = compute_psnr_masked(orig, recon, edge_mask)
    psnr_smooth = compute_psnr_masked(orig, recon, smooth_mask)
    return psnr_edge, psnr_smooth


# --- Load model ---
def load_model(checkpoint_path: str, device: str):
    from scripts.evaluation.evaluate import load_model_from_checkpoint
    model, _, _, _ = load_model_from_checkpoint(checkpoint_path, device)
    return model


# --- Main collection ---
@torch.no_grad()
def collect_metrics(
    models: Dict[str, torch.nn.Module],
    dataloader: DataLoader,
    device: str,
    num_samples: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Tuple[float, float]]]:
    """
    Collect per-frequency and edge/smooth metrics for each model.
    Returns:
        freq_metrics: {model: {band_name: psnr}}
        region_metrics: {model: (psnr_edge, psnr_smooth)}
    """
    freq_metrics = {m: {name: [] for name, _, _ in FREQ_BANDS} for m in models}
    region_edge = {m: [] for m in models}
    region_smooth = {m: [] for m in models}

    samples_seen = 0
    for x, _ in tqdm(dataloader, desc="Evaluating"):
        if samples_seen >= num_samples:
            break
        x = x.to(device=device, dtype=torch.float32)
        B = x.size(0)
        samples_seen += B

        # Denormalize for metrics [0, 1]
        x_denorm = (x * STD.to(device) + MEAN.to(device)).clamp(0, 1)

        for name, model in models.items():
            recon = model(x)[0]
            recon_denorm = (recon * STD.to(device) + MEAN.to(device)).clamp(0, 1)

            # Frequency bands
            for band_name, r_min, r_max in FREQ_BANDS:
                psnr = compute_psnr_band(x_denorm, recon_denorm, r_min, r_max)
                freq_metrics[name][band_name].append(psnr)

            # Edge vs smooth
            pe, ps = compute_psnr_edge_smooth(x_denorm, recon_denorm)
            region_edge[name].append(pe)
            region_smooth[name].append(ps)

    # Aggregate
    freq_agg = {}
    for m in models:
        freq_agg[m] = {band: float(np.mean(v)) for band, v in freq_metrics[m].items() if v}

    region_agg = {}
    for m in models:
        region_agg[m] = (float(np.mean(region_edge[m])), float(np.mean(region_smooth[m])))

    return freq_agg, region_agg


# --- Plotting ---
MODEL_COLORS = {
    "FSQ": "#1f77b4",
    "SimVQ": "#d62728",
    "LMB": "#9467bd",
    "LMB-Fixed": "#8c564b",
    "VQ": "#ff7f0e",
}


def plot_frequency_bands(freq_metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Bar plot: PSNR vs frequency band per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    bands = [b[0] for b in FREQ_BANDS]
    x = np.arange(len(bands))
    width = 0.8 / len(freq_metrics)

    for i, (model, data) in enumerate(freq_metrics.items()):
        vals = [data[FREQ_BANDS[j][0]] for j in range(len(bands))]
        offset = (i - len(freq_metrics) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model, color=MODEL_COLORS.get(model, "#888"))

    ax.set_ylabel("PSNR (dB)")
    ax.set_xlabel("Spatial frequency band")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in FREQ_BANDS])
    ax.legend()
    ax.set_title("Reconstruction quality by frequency band\n(low=structure, high=fine texture)")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_edge_vs_smooth(region_metrics: Dict[str, Tuple[float, float]], output_path: Path):
    """Bar plot: Edge-region PSNR vs Flat-region PSNR per model."""
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(region_metrics.keys())
    edge_psnr = [region_metrics[m][0] for m in models]
    smooth_psnr = [region_metrics[m][1] for m in models]
    x = np.arange(len(models))
    width = 0.35
    ax.bar(x - width / 2, edge_psnr, width, label="Edge regions", color="#e74c3c", alpha=0.9)
    ax.bar(x + width / 2, smooth_psnr, width, label="Smooth regions", color="#3498db", alpha=0.9)
    ax.set_ylabel("PSNR (dB)")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_title("Reconstruction quality: Edge vs Smooth regions")
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Per-frequency and edge/smooth reconstruction analysis")
    parser.add_argument("--data-root", type=str, default="data/imagenet")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default="plots")
    parser.add_argument("--checkpoints", type=str, nargs="+", default=None,
                        help="Override: model:path pairs, e.g. FSQ:path.pt LMB:path.pt")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint paths
    checkpoints = {}
    if args.checkpoints:
        for kv in args.checkpoints:
            k, v = kv.split(":", 1)
            checkpoints[k] = str(Path(v).resolve())
    else:
        for k, v in DEFAULT_CHECKPOINTS.items():
            p = project_root / v
            if p.exists():
                checkpoints[k] = str(p)
            else:
                print(f"Warning: {k} checkpoint not found: {p}")

    if not checkpoints:
        print("No checkpoints found. Use --checkpoints Model:path.pt ...")
        return 1

    # Data: prefer val/ then test/ then train/
    data_path = None
    for split in ("val", "test", "train"):
        p = Path(args.data_root) / split
        if p.exists():
            data_path = p
            break
    if data_path is None:
        print(f"Data not found: {args.data_root} (need val/, test/, or train/)")
        return 1

    dataset = FlatImageDataset(str(data_path), transform=get_transform(128), max_images=args.num_samples)
    if len(dataset) == 0:
        print(f"No images found in {data_path}")
        return 1
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load models
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}
    for name, path in checkpoints.items():
        print(f"Loading {name}...")
        models[name] = load_model(path, device)

    # Collect metrics
    print("Collecting per-frequency and edge/smooth metrics...")
    freq_metrics, region_metrics = collect_metrics(models, loader, device, args.num_samples)

    # Plots
    plot_frequency_bands(freq_metrics, output_dir / "psnr_vs_frequency_band.png")
    plot_edge_vs_smooth(region_metrics, output_dir / "psnr_edge_vs_smooth.png")

    return 0


if __name__ == "__main__":
    sys.exit(main())

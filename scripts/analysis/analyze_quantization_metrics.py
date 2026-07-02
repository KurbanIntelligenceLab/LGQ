#!/usr/bin/env python3
"""
Analyze quantization metrics for trained models.

Computes:
1. Average distance from z → nearest active code
2. Wasserstein distance between z distribution and e distribution
3. Quantization error vs utilization curve
4. Density-weighted code usage

Usage:
    python scripts/analyze_quantization_metrics.py \
        --checkpoint results/fsq/.../best_model.pt \
        --num-images 1000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from scipy.stats import wasserstein_distance

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity


class FlatImageDataset(Dataset):
    """Load images from a directory (including subdirs)."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    def __init__(self, root: str, transform=None, max_images: int = None):
        root = Path(root)
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(root.glob(f"*{ext}"))
            paths.extend(root.glob(f"*{ext.upper()}"))
            paths.extend(root.glob(f"**/*{ext}"))
            paths.extend(root.glob(f"**/*{ext.upper()}"))
        self.paths = sorted(set(paths))
        if max_images:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise RuntimeError(f"No images in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_transform(image_size: int = 128):
    def _transform(img):
        img = img.resize((image_size, image_size), Image.LANCZOS)
        x = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std
    return _transform


def load_model(checkpoint_path: str, device: str):
    from configs import MODEL_CONFIGS
    from quantization.model import ModelConfig

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model_type = ckpt.get("model_type", config.get("model"))
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}")

    model_cfg = ModelConfig(
        base_ch=config.get("dim", 128) // 2,
        embedding_dim=config.get("embedding_dim", 128),
    )
    model = MODEL_CONFIGS[model_type].create_model(argparse.Namespace(**config), model_cfg)

    state_dict = ckpt["model_state_dict"].copy()
    if "quantize.centers" in state_dict:
        centers = state_dict["quantize.centers"]
        if hasattr(model, "quantize") and hasattr(model.quantize, "centers"):
            if model.quantize.centers.shape != centers.shape:
                if centers.shape[0] == model.quantize.centers.shape[1]:
                    state_dict["quantize.centers"] = centers.t()
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model, config, model_type


def _z_norm_unified(model, x: torch.Tensor, model_type: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified function to get z_norm and indices for all model types."""
    z = model.enc(x)
    B, C, H, W = z.shape
    
    if model_type == "lmb":
        # LMB: flatten spatial dims, apply pre_vq_norm
        z_bct = z.view(B, C, H * W)
        z_flat = z_bct.permute(0, 2, 1).contiguous().view(B * (H * W), C)
        z_norm = model.pre_vq_norm(z_flat)
        out = model(x)
        indices = out[1]  # [B,T,C] per-channel or [B,T] flattened
    else:
        # FSQ, VQ, LFQ, SimVQ: standard flattening
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat)
        out = model(x)
        indices = out[1]  # [B,H,W] or [B,T]
    
    return z_norm, indices


@torch.no_grad()
def collect_z_and_active(
    model,
    dataloader: DataLoader,
    device: str,
    model_type: str,
    *,
    num_images: int = 1000,
    subsample_latents: int = 15000,
) -> Tuple[torch.Tensor, int, list]:
    """Collect z_norm and active code indices."""
    model.eval()
    all_z: list = []
    all_idx: list = []
    images_done = 0

    for x, _ in tqdm(dataloader, desc=f"Collecting {model_type}", total=min(len(dataloader), num_images // dataloader.batch_size + 1)):
        if images_done >= num_images:
            break
        x = x.to(device)
        images_done += x.size(0)
        z_norm, idx = _z_norm_unified(model, x, model_type)
        all_z.append(z_norm.cpu())
        all_idx.append(idx.cpu())

    z_full = torch.cat(all_z, dim=0)
    
    # Handle indices based on model type
    if model_type == "lmb":
        config = getattr(model, "config", {})
        lmb_flattened = bool(config.get("flatten_channels", False)) if hasattr(model, "config") else False
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened:
            idx_full = torch.cat([i.reshape(-1, i.size(-1)) for i in all_idx], dim=0)  # [N, C]
        else:
            idx_full = torch.cat([i.reshape(-1) for i in all_idx], dim=0)  # [N]
    else:
        idx_full = torch.cat([i.reshape(-1) for i in all_idx], dim=0)  # [N]
    
    n_total = z_full.size(0)

    if n_total > subsample_latents:
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=rng)[:subsample_latents]
        z = z_full[perm]
        idx_sub = idx_full[perm]
    else:
        z = z_full
        idx_sub = idx_full

    active_data: list
    if model_type == "lmb":
        config = getattr(model, "config", {})
        lmb_flattened = bool(config.get("flatten_channels", False)) if hasattr(model, "config") else False
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened and idx_sub.dim() > 1:
            rows = [tuple(r.tolist()) for r in idx_sub.numpy()]
            unique_tuples = sorted(set(rows))
            active_data = [tuple(t) for t in unique_tuples]
            n_active = len(unique_tuples)
        else:
            flat = idx_sub.reshape(-1).numpy()
            active_set = sorted(set(int(i) for i in flat))
            active_data = active_set
            n_active = len(active_set)
    else:
        flat = idx_sub.reshape(-1).numpy()
        active_set = sorted(set(int(i) for i in flat))
        active_data = active_set
        n_active = len(active_set)

    print(f"  Tokens: {n_total}, Subsampled: {z.size(0)}, Active codes (in subset): {n_active}")
    return z, n_active, active_data


def get_active_code_vectors(
    model,
    model_type: str,
    active_data: list,
    device: str,
) -> torch.Tensor:
    """Build [E, dim] active code vectors in same space as z_norm."""
    if model_type == "lmb":
        config = getattr(model, "config", {})
        lmb_flattened = bool(config.get("flatten_channels", False)) if hasattr(model, "config") else False
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened and isinstance(active_data[0], tuple):
            # Per-channel: active_data list of tuples
            centers = model.quantize.centers.detach().cpu()  # [C, K]
            U = len(active_data)
            C = len(active_data[0])
            e = torch.zeros(U, C, dtype=centers.dtype)
            for u, tuple_idx in enumerate(active_data):
                for c in range(C):
                    e[u, c] = centers[c, tuple_idx[c]]
            return e
        else:
            # LMB flattened: codebook [K, C]
            codebook = model.quantize.centers.detach().cpu()  # [K, C]
            idx = torch.tensor(active_data, dtype=torch.long)
            return codebook[idx]
    elif model_type == "fsq":
        idx = torch.tensor(active_data, dtype=torch.long)
        embedding_dim = model.model_config.embedding_dim
        n_codes = len(idx)
        h = w = int(np.sqrt(n_codes))
        if h * w < n_codes:
            h = w = int(np.ceil(np.sqrt(n_codes)))
        idx_padded = idx[:h*w]
        if len(idx_padded) < h * w:
            padding = torch.zeros(h * w - len(idx_padded), dtype=idx.dtype)
            idx_padded = torch.cat([idx_padded, padding])
        idx_2d = idx_padded.view(1, h, w)
        # Move to same device as model
        model_device = next(model.parameters()).device
        idx_2d = idx_2d.to(model_device)
        codes_quantize_dim = model.quantize.indices_to_codes(idx_2d)
        codes = model.post_quant(codes_quantize_dim)
        codes_flat = codes.permute(0, 2, 3, 1).contiguous().view(h * w, codes.size(1))
        return codes_flat[:n_codes].cpu()
    elif model_type == "lfq":
        idx = torch.tensor(active_data, dtype=torch.long)
        n_codes = len(idx)
        h = w = int(np.sqrt(n_codes))
        if h * w < n_codes:
            h = w = int(np.ceil(np.sqrt(n_codes)))
        idx_padded = idx[:h*w]
        if len(idx_padded) < h * w:
            padding = torch.zeros(h * w - len(idx_padded), dtype=idx.dtype)
            idx_padded = torch.cat([idx_padded, padding])
        idx_2d = idx_padded.view(1, h, w)
        # Move to same device as model
        model_device = next(model.parameters()).device
        idx_2d = idx_2d.to(model_device)
        codes_quantize_dim = model.quantize.indices_to_codes(idx_2d, project_out=False)
        codes = model.post_quant(codes_quantize_dim)
        codes_flat = codes.permute(0, 2, 3, 1).contiguous().view(h * w, codes.size(1))
        return codes_flat[:n_codes].cpu()
    elif model_type in ["vq", "sim_vq"]:
        if model_type == "sim_vq":
            codebook = model.quantize.codebook.detach().cpu()
        else:
            codebook = model.quantize.codebook.detach().cpu()
        idx = torch.tensor(active_data, dtype=torch.long)
        return codebook[idx]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_avg_distance_to_nearest_code(z: torch.Tensor, e: torch.Tensor) -> float:
    """Compute average distance from each z to its nearest active code e."""
    z_np = z.detach().cpu().numpy()
    e_np = e.detach().cpu().numpy()
    
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(e_np)
    distances, _ = nn.kneighbors(z_np)
    
    return float(distances.mean())


def compute_wasserstein_distance(z: torch.Tensor, e: torch.Tensor, n_samples: int = 10000) -> Dict[str, float]:
    """Compute Wasserstein distance between z distribution and e distribution."""
    z_np = z.detach().cpu().numpy()
    e_np = e.detach().cpu().numpy()
    
    # Subsample if needed
    if z_np.shape[0] > n_samples:
        z_idx = np.random.choice(z_np.shape[0], n_samples, replace=False)
        z_np = z_np[z_idx]
    if e_np.shape[0] > n_samples:
        e_idx = np.random.choice(e_np.shape[0], n_samples, replace=False)
        e_np = e_np[e_idx]
    
    dim = z_np.shape[1]
    wasserstein_per_dim = []
    
    for d in range(dim):
        z_dim = z_np[:, d]
        e_dim = e_np[:, d]
        wd = wasserstein_distance(z_dim, e_dim)
        wasserstein_per_dim.append(wd)
    
    wasserstein_per_dim = np.array(wasserstein_per_dim)
    
    return {
        "mean_wasserstein": float(wasserstein_per_dim.mean()),
        "std_wasserstein": float(wasserstein_per_dim.std()),
        "min_wasserstein": float(wasserstein_per_dim.min()),
        "max_wasserstein": float(wasserstein_per_dim.max()),
    }


def compute_quantization_error_vs_utilization(
    z: torch.Tensor,
    e: torch.Tensor,
    indices: torch.Tensor,
    codebook_size: int,
    n_bins: int = 20,
) -> Dict:
    """Compute quantization error vs codebook utilization curve."""
    z_np = z.detach().cpu().numpy()
    e_np = e.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy().flatten()
    
    # Subsample indices to match z size
    n_z = z_np.shape[0]
    if len(indices_np) > n_z:
        # Take first n_z indices (they correspond to same samples as z)
        indices_np = indices_np[:n_z]
    
    # Compute code usage counts
    unique_indices, counts = np.unique(indices_np, return_counts=True)
    code_usage = np.zeros(codebook_size)
    for idx, cnt in zip(unique_indices, counts):
        if idx < codebook_size:
            code_usage[idx] = cnt
    
    # Compute utilization
    active_codes = (code_usage > 0).sum()
    utilization = 100.0 * active_codes / codebook_size
    
    # For each z, find nearest e and compute distance
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(e_np)
    distances, _ = nn.kneighbors(z_np)
    distances = distances.flatten()
    
    # Map indices to e positions (simplified - assumes indices match)
    # Group errors by code usage frequency
    code_to_usage = {idx: count for idx, count in zip(unique_indices, counts)}
    
    # Bin by utilization percentage
    utilization_bins = np.linspace(0, 100, n_bins + 1)
    bin_errors = [[] for _ in range(n_bins)]
    
    # For each z, get its assigned code and error (both now have same length)
    for i, idx in enumerate(indices_np):
        if i >= len(distances):
            break
        if idx in code_to_usage:
            usage_pct = 100.0 * code_to_usage[idx] / counts.sum() if counts.sum() > 0 else 0.0
            bin_idx = np.digitize(usage_pct, utilization_bins) - 1
            bin_idx = max(0, min(bin_idx, n_bins - 1))
            bin_errors[bin_idx].append(distances[i])
    
    # Compute statistics per bin
    avg_errors = []
    std_errors = []
    bin_centers = []
    
    for i in range(n_bins):
        if len(bin_errors[i]) > 0:
            avg_errors.append(np.mean(bin_errors[i]))
            std_errors.append(np.std(bin_errors[i]))
            bin_centers.append((utilization_bins[i] + utilization_bins[i + 1]) / 2)
        else:
            avg_errors.append(np.nan)
            std_errors.append(np.nan)
            bin_centers.append((utilization_bins[i] + utilization_bins[i + 1]) / 2)
    
    return {
        "utilization_bins": np.array(bin_centers),
        "avg_quant_error": np.array(avg_errors),
        "std_quant_error": np.array(std_errors),
        "overall_utilization": utilization,
    }


def compute_density_weighted_code_usage(
    z: torch.Tensor,
    e: torch.Tensor,
    indices: torch.Tensor,
    bandwidth: float = 0.1,
) -> Dict[str, float]:
    """Compute density-weighted code usage."""
    z_np = z.detach().cpu().numpy()
    e_np = e.detach().cpu().numpy()
    indices_np = indices.detach().cpu().numpy().flatten()
    
    # Subsample z for KDE if too large
    max_samples = 10000
    if z_np.shape[0] > max_samples:
        z_idx = np.random.choice(z_np.shape[0], max_samples, replace=False)
        z_sub = z_np[z_idx]
    else:
        z_sub = z_np
    
    # Fit KDE on z
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(z_sub)
    
    # Compute density at each e location
    e_densities = np.exp(kde.score_samples(e_np))
    
    # Compute actual code usage
    unique_indices, counts = np.unique(indices_np, return_counts=True)
    code_usage = np.zeros(len(e_np))
    for idx, count in zip(unique_indices, counts):
        if idx < len(e_np):
            code_usage[idx] = count
    
    # Normalize
    e_densities_norm = e_densities / (e_densities.sum() + 1e-10)
    code_usage_norm = code_usage / (code_usage.sum() + 1e-10)
    
    # Compute correlation between density and usage
    correlation = np.corrcoef(e_densities_norm, code_usage_norm)[0, 1]
    
    # Compute density-weighted usage
    density_weighted_usage = (code_usage_norm * e_densities_norm).sum()
    
    return {
        "density_weighted_usage": float(density_weighted_usage),
        "density_usage_correlation": float(correlation),
        "mean_code_density": float(e_densities.mean()),
        "std_code_density": float(e_densities.std()),
    }


@torch.no_grad()
def collect_data_for_analysis(
    model,
    dataloader: DataLoader,
    device: str,
    model_type: str,
    num_images: int = 1000,
    subsample_latents: int = 15000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Collect z, e, and indices for analysis."""
    # Collect z and active indices
    z, n_active, active_data = collect_z_and_active(
        model, dataloader, device, model_type,
        num_images=num_images,
        subsample_latents=subsample_latents,
    )
    
    # Get active code vectors
    e = get_active_code_vectors(model, model_type, active_data, device)
    
    # Get all indices (not just active ones) for utilization analysis
    all_indices = []
    images_done = 0
    for x, _ in tqdm(dataloader, desc="Collecting indices", total=min(len(dataloader), num_images // dataloader.batch_size + 1)):
        if images_done >= num_images:
            break
        x = x.to(device)
        images_done += x.size(0)
        _, idx = _z_norm_unified(model, x, model_type)
        all_indices.append(idx.cpu())
    
    # Flatten indices
    if model_type == "lmb":
        config = getattr(model, "config", {})
        lmb_flattened = bool(config.get("flatten_channels", False)) if hasattr(model, "config") else False
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened:
            indices_flat = torch.cat([i.reshape(-1, i.size(-1)) for i in all_indices], dim=0)
            indices_flat = indices_flat[:, 0] if indices_flat.dim() > 1 else indices_flat.reshape(-1)
        else:
            indices_flat = torch.cat([i.reshape(-1) for i in all_indices], dim=0)
    else:
        indices_flat = torch.cat([i.reshape(-1) for i in all_indices], dim=0)
    
    # Get codebook size
    if model_type == "lmb":
        if hasattr(model.quantize, "codebook_size"):
            codebook_size = model.quantize.codebook_size
        elif hasattr(model.quantize, "K"):
            codebook_size = model.quantize.K
        else:
            codebook_size = len(active_data) if isinstance(active_data, list) else 1000
    elif model_type == "fsq":
        codebook_size = model.num_codes
    elif model_type in ["vq", "sim_vq"]:
        codebook_size = model.quantize.codebook_size
    elif model_type == "lfq":
        codebook_size = 2 ** model.quantize_dim
    else:
        codebook_size = 1000
    
    return z, e, indices_flat, codebook_size


def plot_quantization_error_vs_utilization(
    results: Dict,
    output_path: Path,
):
    """Plot quantization error vs utilization curve."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    bins = results["utilization_bins"]
    avg_errors = results["avg_quant_error"]
    std_errors = results["std_quant_error"]
    
    # Filter out NaN values
    valid_mask = ~np.isnan(avg_errors)
    bins = bins[valid_mask]
    avg_errors = avg_errors[valid_mask]
    std_errors = std_errors[valid_mask]
    
    ax.plot(bins, avg_errors, 'o-', label='Mean quantization error', linewidth=2, markersize=6)
    ax.fill_between(bins, avg_errors - std_errors, avg_errors + std_errors, alpha=0.3, label='±1 std')
    
    ax.set_xlabel('Codebook Utilization (%)', fontsize=12)
    ax.set_ylabel('Quantization Error (L2 distance)', fontsize=12)
    ax.set_title('Quantization Error vs Codebook Utilization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze quantization metrics")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--data-root", type=str, default="data/imagenet", help="Data root directory")
    parser.add_argument("--num-images", type=int, default=1000, help="Number of images to process")
    parser.add_argument("--subsample-latents", type=int, default=15000, help="Subsample latents for efficiency")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--output-dir", type=str, default="results/plots", help="Output directory")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, config, model_type = load_model(args.checkpoint, device)
    print(f"Model type: {model_type}")
    
    # Load data
    data_path = Path(args.data_root).expanduser()
    test_path = data_path / "test"
    if not test_path.exists():
        test_path = data_path / "val"
    
    print(f"\nLoading images from: {test_path}")
    transform = get_transform(image_size=128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Collect data
    print(f"\nCollecting data for analysis...")
    z, e, indices, codebook_size = collect_data_for_analysis(
        model, dataloader, device, model_type,
        num_images=args.num_images,
        subsample_latents=args.subsample_latents,
    )
    
    print(f"  Collected {z.shape[0]} latent vectors")
    print(f"  Found {e.shape[0]} active codes")
    print(f"  Codebook size: {codebook_size}")
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing metrics...")
    print(f"{'='*60}")
    
    # 1. Average distance from z → nearest active code
    print("\n1. Computing average distance from z → nearest active code...")
    avg_dist = compute_avg_distance_to_nearest_code(z, e)
    print(f"   Average distance: {avg_dist:.4f}")
    
    # 2. Wasserstein distance
    print("\n2. Computing Wasserstein distance between z and e distributions...")
    wasserstein_metrics = compute_wasserstein_distance(z, e)
    print(f"   Mean Wasserstein distance: {wasserstein_metrics['mean_wasserstein']:.4f}")
    print(f"   Std Wasserstein distance: {wasserstein_metrics['std_wasserstein']:.4f}")
    print(f"   Min: {wasserstein_metrics['min_wasserstein']:.4f}, Max: {wasserstein_metrics['max_wasserstein']:.4f}")
    
    # 3. Quantization error vs utilization
    print("\n3. Computing quantization error vs utilization curve...")
    error_vs_util = compute_quantization_error_vs_utilization(z, e, indices, codebook_size)
    print(f"   Overall utilization: {error_vs_util['overall_utilization']:.2f}%")
    valid_errors = error_vs_util['avg_quant_error'][~np.isnan(error_vs_util['avg_quant_error'])]
    if len(valid_errors) > 0:
        print(f"   Mean quantization error: {valid_errors.mean():.4f}")
    
    # 4. Density-weighted code usage
    print("\n4. Computing density-weighted code usage...")
    density_metrics = compute_density_weighted_code_usage(z, e, indices)
    print(f"   Density-weighted usage: {density_metrics['density_weighted_usage']:.4f}")
    print(f"   Density-usage correlation: {density_metrics['density_usage_correlation']:.4f}")
    
    # Save results
    results = {
        "model_type": model_type,
        "checkpoint": args.checkpoint,
        "num_images": args.num_images,
        "num_latents": int(z.shape[0]),
        "num_active_codes": int(e.shape[0]),
        "codebook_size": int(codebook_size),
        "avg_distance_to_nearest_code": avg_dist,
        "wasserstein_distance": wasserstein_metrics,
        "quantization_error_vs_utilization": error_vs_util,
        "density_weighted_code_usage": density_metrics,
    }
    
    # Save text summary
    output_file = output_dir / f"quantization_metrics_{model_type}.txt"
    with open(output_file, 'w') as f:
        f.write("Quantization Metrics Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Images processed: {args.num_images}\n")
        f.write(f"Latent vectors: {z.shape[0]}\n")
        f.write(f"Active codes: {e.shape[0]}\n")
        f.write(f"Codebook size: {codebook_size}\n\n")
        
        f.write("1. Average distance from z → nearest active code:\n")
        f.write(f"   {avg_dist:.6f}\n\n")
        
        f.write("2. Wasserstein distance (z distribution vs e distribution):\n")
        for k, v in wasserstein_metrics.items():
            f.write(f"   {k}: {v:.6f}\n")
        f.write("\n")
        
        f.write("3. Quantization error vs utilization:\n")
        f.write(f"   Overall utilization: {error_vs_util['overall_utilization']:.2f}%\n")
        if len(valid_errors) > 0:
            f.write(f"   Mean quantization error: {valid_errors.mean():.6f}\n")
        f.write("\n")
        
        f.write("4. Density-weighted code usage:\n")
        for k, v in density_metrics.items():
            f.write(f"   {k}: {v:.6f}\n")
        f.write("\n")
    
    print(f"\nSaved results to: {output_file}")
    
    # Generate plots
    if not args.no_plots:
        print(f"\nGenerating plots...")
        plot_path = output_dir / f"quantization_error_vs_utilization_{model_type}.png"
        plot_quantization_error_vs_utilization(error_vs_util, plot_path)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

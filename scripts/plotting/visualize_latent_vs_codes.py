#!/usr/bin/env python3
"""
Visualize encoder outputs (z) vs active codebook entries (e) using UMAP and t-SNE.

Compares all quantization methods: FSQ, VQ, LFQ, SimVQ, LMB (flattened), and LMB per-channel fair.
Append z to e, run dimensionality reduction, plot z as dots and e as X markers.
If codes cluster more tightly around encoder outputs, they better represent
the latent distribution.

Usage:
    # Main experiments (all models 16K), latest epoch:
    python scripts/visualize_latent_vs_codes.py --main

    # Custom checkpoints:
    python scripts/visualize_latent_vs_codes.py \\
        --checkpoints results/fsq/.../latest_model.pt \\
                     results/vq/.../latest_model.pt \\
                     results/lfq/.../latest_model.pt \\
                     results/sim_vq/.../latest_model.pt \\
                     results/lmb/.../latest_model.pt \\
        --num-images 1000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Default main-experiment paths (latest epoch, 16K codebook)
MAIN_CHECKPOINTS = {
    "fsq": "results/fsq/fsq_lv16-16-8-8_bs32_lr3e-4_dim128_20260123_184222_deab/checkpoints/latest_model.pt",
    "vq": "results/vq/vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0/checkpoints/latest_model.pt",
    "lfq": "results/lfq/lfq_cb16384_bs32_lr3e-4_dim128_20260123_221044_6f79/checkpoints/latest_model.pt",
    "sim_vq": "results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32/checkpoints/latest_model.pt",
    "lmb": "results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd/checkpoints/latest_model.pt",  # flattened 16K
    "lmb_fair": "results/lmb/lmb_ablation_perchannel_fair/checkpoints/latest_model.pt",  # LMB per-channel fair (16K)
}


def get_last_epoch_checkpoint(ckpt_path: Path) -> Path:
    """If checkpoint dir has checkpoint_epoch_*.pt, return the one with highest epoch; else return ckpt_path."""
    ckpt_dir = ckpt_path.parent
    if not ckpt_dir.is_dir():
        return ckpt_path
    epoch_files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if not epoch_files:
        return ckpt_path
    last_epoch = -1
    chosen = ckpt_path
    for f in epoch_files:
        try:
            # checkpoint_epoch_035.pt -> 35
            epoch = int(f.stem.rsplit("_", 1)[-1])
            if epoch > last_epoch:
                last_epoch = epoch
                chosen = f
        except (ValueError, IndexError):
            continue
    return chosen if last_epoch >= 0 else ckpt_path


def discover_lmb_fair_checkpoint() -> Path | None:
    """Discover LMB per-channel fair checkpoint under results/lmb (dir name contains 'perchannel_fair')."""
    lmb_root = project_root / "results" / "lmb"
    if not lmb_root.is_dir():
        return None
    for exp_dir in sorted(lmb_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        if "perchannel_fair" not in exp_dir.name:
            continue
        ckpt = exp_dir / "checkpoints" / "latest_model.pt"
        if ckpt.exists():
            return ckpt
    return None


class FlatImageDataset(Dataset):
    """Load images from a directory (including subdirs)."""
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    def __init__(self, root: str, transform=None, max_images: int | None = None):
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


def _z_norm_unified(model, x: torch.Tensor, model_type: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Unified function to get z and indices for all model types.
    For FSQ/LFQ: returns z in quantize_dim (after pre_quant) so it matches the space where codes live.
    For others: returns z_norm in embedding_dim."""
    z = model.enc(x)
    B, C, H, W = z.shape
    
    if model_type == "lmb":
        if model.pre_quant is not None:
            # LMB per-channel fair: quantization in quantize_dim (after pre_quant), same as FSQ
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_quant = model.pre_quant(z_norm)  # [B, quantize_dim, H, W]
            z_quant_flat = z_quant.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)
            out = model(x)
            indices = out[1]  # [B, T, C]
            return z_quant_flat, indices
        # LMB flattened or standard per-channel: quantization in embedding_dim
        z_bct = z.view(B, C, H * W)
        z_flat = z_bct.permute(0, 2, 1).contiguous().view(B * (H * W), C)
        z_norm = model.pre_vq_norm(z_flat)
        out = model(x)
        indices = out[1]  # [B,T,C] per-channel or [B,T] flattened
        return z_norm, indices
    elif model_type in ("fsq", "lfq"):
        # FSQ/LFQ: quantization happens in quantize_dim (after pre_quant). Return z in that space.
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        z_quant = model.pre_quant(z_norm)  # [B, quantize_dim, H, W]
        z_quant_flat = z_quant.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)
        out = model(x)
        indices = out[1]  # [B, H, W]
        return z_quant_flat, indices
    else:
        # VQ, SimVQ: standard flattening
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
) -> tuple[torch.Tensor, int, list | np.ndarray]:
    """
    Collect z_norm (encoder outputs after pre_vq_norm) and active code indices.
    Subsamples z and indices together, then computes unique active codes from
    the subsampled set only (same subset for fair metrics).
    Returns (z [N,dim], n_active, active_indices_or_vectors).
    """
    model.eval()
    all_z: list[torch.Tensor] = []
    all_idx: list[torch.Tensor] = []
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
        # Try to infer from model structure
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened:
            # Per-channel: indices are [B, T, C] or [B, H, W, C]
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

    active_data: list | np.ndarray
    if model_type == "lmb":
        config = getattr(model, "config", {})
        lmb_flattened = bool(config.get("flatten_channels", False)) if hasattr(model, "config") else False
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened and idx_sub.dim() > 1:
            # Per-channel: active_data [U, C]
            rows = [tuple(r.tolist()) for r in idx_sub.numpy()]
            unique_tuples = sorted(set(rows))
            active_data = np.array(unique_tuples, dtype=np.int64)
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
    return z, n_active, active_data, idx_sub


def get_active_code_vectors(
    model,
    model_type: str,
    active_data: list | np.ndarray,
    device: str,
) -> torch.Tensor:
    """Build [E, dim] active code vectors in same space as z_norm."""
    if model_type == "lmb":
        # Check if per-channel or flattened
        config = getattr(model, "config", {})
        lmb_flattened = bool(config.get("flatten_channels", False)) if hasattr(model, "config") else False
        if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
            lmb_flattened = model.quantize.flatten_channels
        
        if not lmb_flattened and isinstance(active_data, np.ndarray) and active_data.ndim > 1:
            # Per-channel: active_data [U, C]. Code u: [centers[c, active_data[u,c]] for c]
            centers = model.quantize.centers.detach().cpu()  # [C, K]
            U, C = active_data.shape
            e = torch.zeros(U, C, dtype=centers.dtype)
            for u in range(U):
                for c in range(C):
                    e[u, c] = centers[c, active_data[u, c]]
            return e
        else:
            # LMB flattened: codebook [K, C]
            codebook = model.quantize.centers.detach().cpu()  # [K, C]
            idx = torch.tensor(active_data, dtype=torch.long)
            return codebook[idx]
    elif model_type == "fsq":
        # FSQ: Keep z and e in quantize_dim (same space where quantization happens).
        # indices_to_codes() applies project_out (embedding_dim); use _indices_to_codes so
        # codes stay in codebook_dim to match z_quant_flat.
        idx = torch.tensor(active_data, dtype=torch.long).to(device)  # [n_codes]
        codes_quantize_dim = model.quantize._indices_to_codes(idx)  # [n_codes, codebook_dim]
        return codes_quantize_dim.cpu()
    elif model_type == "lfq":
        # LFQ: Keep z and e in quantize_dim (same space where quantization happens).
        # Pass 1D indices; indices_to_codes(..., project_out=False) returns [n_codes, quantize_dim].
        idx = torch.tensor(active_data, dtype=torch.long).to(device)  # [n_codes]
        codes_quantize_dim = model.quantize.indices_to_codes(idx, project_out=False)  # [n_codes, quantize_dim]
        return codes_quantize_dim.cpu()
    elif model_type in ["vq", "rot_vq", "sim_vq"]:
        # VQ, RotVQ, SimVQ: standard codebook lookup
        codebook = model.quantize.codebook.detach().cpu()  # [K, C]
        idx = torch.tensor(active_data, dtype=torch.long)
        return codebook[idx]
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def run_umap(
    z: torch.Tensor,
    e: torch.Tensor,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    standardize: bool = False,
):
    """Run UMAP on combined z and e. Optional joint standardization helps FSQ/LFQ when z and e differ in scale."""
    import umap
    combined = torch.cat([z, e], dim=0).detach().numpy()
    if standardize:
        mean = combined.mean(axis=0)
        std = combined.std(axis=0)
        std[std < 1e-8] = 1.0
        combined = (combined - mean) / std
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="euclidean", random_state=42, verbose=False)
    emb = reducer.fit_transform(combined)
    nz = z.size(0)
    z_emb = emb[:nz]
    e_emb = emb[nz:]
    return z_emb, e_emb


def run_tsne(z: torch.Tensor, e: torch.Tensor, perplexity: float = 30.0, n_iter: int = 1000):
    from sklearn.manifold import TSNE
    import os
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    # Subsample if too large to avoid memory issues
    max_samples = 10000
    nz = z.size(0)
    ne = e.size(0)
    if nz + ne > max_samples:
        # Subsample proportionally
        z_ratio = nz / (nz + ne)
        z_samples = int(max_samples * z_ratio)
        e_samples = max_samples - z_samples
        z_idx = torch.randperm(nz)[:z_samples]
        e_idx = torch.randperm(ne)[:e_samples]
        z_sub = z[z_idx]
        e_sub = e[e_idx]
    else:
        z_sub = z
        e_sub = e
        z_idx = None
        e_idx = None
    
    combined = torch.cat([z_sub, e_sub], dim=0).detach().cpu().numpy().astype(np.float32)
    # Reduce perplexity if dataset is too small
    n_samples = combined.shape[0]
    actual_perplexity = min(perplexity, (n_samples - 1) / 3, 30.0)
    reducer = TSNE(n_components=2, perplexity=actual_perplexity, max_iter=min(n_iter, 500), random_state=42, verbose=0, n_jobs=1, method='barnes_hut', angle=0.5)
    emb = reducer.fit_transform(combined)
    nz_sub = z_sub.size(0)
    z_emb = emb[:nz_sub]
    e_emb = emb[nz_sub:]
    return z_emb, e_emb


def compute_metrics(z: torch.Tensor, e: torch.Tensor):
    from sklearn.neighbors import NearestNeighbors
    z_np = z.detach().numpy() if isinstance(z, torch.Tensor) else z
    e_np = e.detach().numpy() if isinstance(e, torch.Tensor) else e
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(e_np)
    d, _ = nn.kneighbors(z_np)
    latent_to_code = d.flatten()
    nn2 = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(z_np)
    d2, _ = nn2.kneighbors(e_np)
    code_to_latent = d2.flatten()
    z_std = float(np.std(z_np))
    orphan = 100.0 * (code_to_latent > 2 * latent_to_code.mean()).mean()
    return {
        "latent_to_code_mean": float(latent_to_code.mean()),
        "relative_quant_error": float(latent_to_code.mean() / z_std),
        "orphan_codes_pct": float(orphan),
    }


def plot_all_models(
    model_data: Dict[str, Dict],
    out_path: Path,
    method: str = "UMAP",
    *,
    show_metrics_in_title: bool = True,
    red_alpha_bump: float = 0.0,
    subsample_red: bool = True,
) -> None:
    """Plot all models in a grid layout. Red (e) drawn on top, clearly visible.
    Encoder output (z) is shown only as a KDE contour (no blue dots)."""
    model_order = ["fsq", "rot_vq", "sim_vq", "lmb", "vq", "ibq", "lfq", "lmb_fair"]
    available_models = [m for m in model_order if m in model_data]
    n_models = len(available_models)

    if n_models == 0:
        raise ValueError("No model data provided")

    # Layout: 6 models = 2x3; 5 models = 2 rows with bottom two centered; else single row
    if n_models == 6:
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
        gs = GridSpec(2, 6, figure=fig, hspace=0.06, wspace=0.06)
        axes_list = [
            fig.add_subplot(gs[0, 0:2]),  # FSQ
            fig.add_subplot(gs[0, 2:4]),  # VQ
            fig.add_subplot(gs[0, 4:6]),  # LFQ
            fig.add_subplot(gs[1, 0:2]),  # SimVQ
            fig.add_subplot(gs[1, 2:4]),  # LMB
            fig.add_subplot(gs[1, 4:6]),  # LMB per-channel fair
        ]
    elif n_models == 5:
        fig = plt.figure(figsize=(14, 8), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
        gs = GridSpec(2, 6, figure=fig, hspace=0.06, wspace=0.06)
        # Top row: 3 plots in cols 0-2, 2-4, 4-6
        axes_list = [
            fig.add_subplot(gs[0, 0:2]),  # FSQ
            fig.add_subplot(gs[0, 2:4]),  # VQ
            fig.add_subplot(gs[0, 4:6]),  # LFQ
            fig.add_subplot(gs[1, 1:3]),  # SimVQ (centered)
            fig.add_subplot(gs[1, 3:5]),  # LMB (centered)
        ]
    elif n_models == 4:
        fig = plt.figure(figsize=(10, 9), constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
        gs = GridSpec(2, 2, figure=fig, hspace=0.06, wspace=0.06)
        axes_list = [
            fig.add_subplot(gs[0, 0]),  # FSQ
            fig.add_subplot(gs[0, 1]),  # RotVQ
            fig.add_subplot(gs[1, 0]),  # SimVQ
            fig.add_subplot(gs[1, 1]),  # LGQ
        ]
    else:
        n_rows = 1
        n_cols = n_models
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5))
        if n_models == 1:
            axes_list = [axes]
        elif n_rows == 1:
            axes_list = axes.tolist() if isinstance(axes, np.ndarray) else list(axes) if hasattr(axes, "__iter__") else [axes]
        else:
            axes_list = axes.flatten().tolist() if isinstance(axes, np.ndarray) else list(axes.flatten())

    plot_idx = 0
    for model_type in available_models:
        data = model_data[model_type]
        ax = axes_list[plot_idx]
        plot_idx += 1

        z_emb = data["z_emb"]
        e_emb = data["e_emb"]
        metrics = data["metrics"]
        n_active = data["n_active"]
        name = data["name"]

        # Encoder output (z) as KDE contour only (no blue dots)
        all_xy = np.vstack([z_emb[:, :2], e_emb[:, :2]])
        xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
        ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
        pad_x = 0.08 * (xmax - xmin) if xmax > xmin else 0.5
        pad_y = 0.08 * (ymax - ymin) if ymax > ymin else 0.5
        xx = np.linspace(xmin - pad_x, xmax + pad_x, 120)
        yy = np.linspace(ymin - pad_y, ymax + pad_y, 120)
        grid = np.meshgrid(xx, yy)
        pts = np.vstack([g.ravel() for g in grid])
        used_kde = False
        try:
            kde = gaussian_kde(z_emb.T, bw_method="scott")
            Z = kde(pts).reshape(grid[0].shape)
            q = np.quantile(Z[Z > 0], [0.02, 0.2, 0.4, 0.6, 0.8, 0.98])
            levels = np.sort(np.unique(np.r_[q, Z.min(), Z.max()]))
            levels = levels[levels >= 1e-10 * Z.max()]
            if len(levels) < 2:
                levels = np.linspace(Z.min(), Z.max(), 8)
            cf = ax.contourf(
                xx, yy, Z, levels=levels, cmap="Blues", alpha=0.72,
                zorder=1, extend="neither",
            )
            for c in cf.collections:
                c.set_rasterized(True)
            # Bold outline so the encoder distribution reads clearly
            ax.contour(
                xx, yy, Z, levels=[levels[0] if len(levels) > 0 else Z.min()],
                colors=["#1a5276"], linewidths=1.2, zorder=1,
            )
            used_kde = True
        except Exception:
            pass  # No fallback to blue dots; show only red active codes

        max_red = 3200
        e_plot = e_emb
        e_subsampled = False
        if subsample_red and e_emb.shape[0] > max_red:
            rng = np.random.default_rng(42)
            idx = rng.choice(e_emb.shape[0], size=max_red, replace=False)
            e_plot = e_emb[idx]
            e_subsampled = True

        if n_active > 7000:
            s_red, alpha_red = 5, 0.35
        elif n_active > 4500:
            s_red, alpha_red = 6, 0.42
        else:
            s_red, alpha_red = 10, 0.55
        alpha_red = min(1.0, alpha_red + red_alpha_bump)

        lbl = "Active codes (e)" if not e_subsampled else f"Active codes (e), n={max_red} of {n_active:,}"
        sc = ax.scatter(
            e_plot[:, 0], e_plot[:, 1],
            c="#e74c3c", s=s_red, alpha=alpha_red, marker="o", label=lbl,
            rasterized=True, edgecolors="#8B0000", linewidths=0.3, zorder=2,
        )
        if show_metrics_in_title:
            err, orphan = metrics["relative_quant_error"], metrics["orphan_codes_pct"]
            title = f"{name}\nActive: {n_active:,}  |  Rel. quant error: {err:.3f}  |  Orphan %: {orphan:.1f}"
        else:
            title = name
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"{method} 1")
        ax.set_ylabel(f"{method} 2")
        if used_kde:
            blue_patch = mpatches.Patch(facecolor=plt.cm.Blues(0.55), alpha=0.72, edgecolor="none", label="Encoder output (z)")
            ax.legend(handles=[blue_patch, sc], loc="upper right", fontsize=9)
        else:
            ax.legend(loc="upper right", fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")

    for idx in range(n_models, len(axes_list)):
        axes_list[idx].axis("off")

    if n_models not in (5, 6):
        plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Visualize z vs active codes (UMAP/t-SNE) for all models")
    ap.add_argument("--main", action="store_true", help="Use main checkpoints for all models (16K)")
    ap.add_argument("--checkpoints", nargs="+", type=str, default=[], help="Custom checkpoint paths (order: fsq, vq, lfq, sim_vq, lmb)")
    ap.add_argument("--data-root", type=str, default="data/imagenet")
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--subsample-latents", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4, help="DataLoader workers. Use 0 to avoid /tmp when disk full.")
    ap.add_argument("--output-dir", type=str, default="plots")
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--tsne-perplexity", type=float, default=30.0)
    ap.add_argument("--tsne-n-iter", type=int, default=1000)
    ap.add_argument("--metrics-only", action="store_true",
                    help="Only compute and save metrics (no UMAP/t-SNE/plots). Fast for large --num-images.")
    ap.add_argument("--skip-tsne", action="store_true", help="Skip t-SNE (run UMAP only).")
    ap.add_argument("--models", nargs="+", type=str, default=None,
                    help="Restrict to these models (e.g. sim_vq lmb). Default: all in main checkpoints.")
    ap.add_argument("--main-three", action="store_true",
                    help="Use only VQ, SimVQ, LMB (side-by-side). Same as --models vq sim_vq lmb.")
    ap.add_argument("--output-suffix", type=str, default="",
                    help="Suffix for metrics filename, e.g. 5k -> latent_vs_codes_all_metrics_5k.txt")
    args = ap.parse_args()
    if args.main_three:
        args.models = ["vq", "sim_vq", "lmb"]

    # Reproducibility: same data order and UMAP/t-SNE every run
    torch.manual_seed(42)
    np.random.seed(42)

    # Determine checkpoints (use last epoch checkpoint per model when available)
    if args.main:
        checkpoints_raw = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
        lmb_fair_path = discover_lmb_fair_checkpoint()
        if lmb_fair_path is not None:
            checkpoints_raw["lmb_fair"] = lmb_fair_path
        if args.models:
            checkpoints_raw = {k: v for k, v in checkpoints_raw.items() if k in args.models}
        checkpoints = {}
        print("Using main experiments (last epoch per model, 16K codebook):")
        for k, v in checkpoints_raw.items():
            resolved = get_last_epoch_checkpoint(v)
            checkpoints[k] = resolved
            label = " (last epoch)" if resolved != v else ""
            print(f"  {k.upper()}: {resolved.name}{label}")
            if not resolved.exists():
                print(f"    ⚠️  Warning: Checkpoint not found: {resolved}")
    else:
        if len(args.checkpoints) == 0:
            raise SystemExit("Provide --checkpoints or use --main")
        checkpoints = {}
        for ckpt_path in args.checkpoints:
            ckpt = Path(ckpt_path)
            if not ckpt.is_absolute():
                ckpt = project_root / ckpt
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)
            mtype = raw.get("model_type") or raw.get("config", {}).get("model")
            if mtype is None:
                raise SystemExit(f"Could not detect model_type from {ckpt}")
            del raw  # free memory
            checkpoints[mtype] = ckpt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Mirror train.py: cuDNN off by default — module CUDA toolkit lacks engines for
    # some conv shapes on this cluster and segfaults inside Conv2d.forward.
    # Override with VQ_CUDNN_ENABLED=1 to re-enable.
    if torch.cuda.is_available():
        import os
        cudnn_on = os.environ.get("VQ_CUDNN_ENABLED", "0") == "1"
        torch.backends.cudnn.enabled = cudnn_on
        if cudnn_on:
            torch.backends.cudnn.benchmark = True
        print(f"cuDNN: enabled={cudnn_on}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser()
    test_path = data_root / "test"
    if not test_path.exists():
        test_path = data_root / "val"
    transform = get_transform(128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(42),
    )

    # Process all models
    all_model_data: Dict[str, Dict] = {}
    
    for model_type, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"⚠️  Skipping {model_type}: checkpoint not found: {ckpt_path}")
            continue
        
        print("\n" + "=" * 60)
        print(f"{model_type.upper()} (16K)")
        print("=" * 60)
        
        try:
            model, config, detected_type = load_model(str(ckpt_path), device)
            # lmb_fair uses same model type as lmb for quantizer logic
            effective_type = detected_type if model_type == "lmb_fair" else model_type
            if effective_type != detected_type and model_type != "lmb_fair":
                print(f"  ⚠️  Warning: Expected {model_type}, got {detected_type}")
            
            z, n_active, active_data, idx_sub = collect_z_and_active(
                model, loader, device, effective_type,
                num_images=args.num_images,
                subsample_latents=args.subsample_latents,
            )
            # FSQ: put z in same bounded [-1,1] space as e (FSQ codes). FSQ.bound() applies
            # the same tanh/scale as the quantizer so z and e are comparable.
            if effective_type == "fsq":
                z = model.quantize.bound(z.to(device)).cpu()
            # FSQ/LFQ indices_to_codes require same device as model; use CPU to avoid mismatch.
            code_device = "cpu"
            if code_device != device:
                model = model.cpu()
            e = get_active_code_vectors(model, effective_type, active_data, code_device)
            # LFQ: raw z (continuous) and e (discrete {-1,1}^14) are in different value spaces,
            # causing UMAP to separate them. Use quantized z (z_q) instead—same space as e—
            # so the plot shows how well the active codes cover the quantized encoder output.
            if effective_type == "lfq":
                idx_t = idx_sub.to(code_device).unsqueeze(0)  # [1, N]
                z_q = model.quantize.indices_to_codes(idx_t, project_out=False).squeeze(0)
                # indices_to_codes with channel_first returns [C, N]; we need [N, C]
                if z_q.dim() == 2 and z_q.shape[0] < z_q.shape[1]:
                    z_q = z_q.T
                z = z_q
            metrics = compute_metrics(z, e)
            
            # Determine model name
            if model_type == "lmb_fair":
                name = "LMB 16K per-channel fair"
            elif model_type == "lmb":
                lmb_flattened = bool(config.get("flatten_channels", False))
                if hasattr(model, "quantize") and hasattr(model.quantize, "flatten_channels"):
                    lmb_flattened = model.quantize.flatten_channels
                name = "LGQ (Ours)" if lmb_flattened else "LMB 128 per-channel"
            else:
                name_map = {
                    "fsq": "FSQ 16K",
                    "vq": "VQ 16K",
                    "lfq": "LFQ 16K",
                    "sim_vq": "SimVQ 16K",
                    "rot_vq": "RotVQ 16K",
                    "ibq": "IBQ 16K",
                }
                name = name_map.get(model_type, model_type.upper())
            
            all_model_data[model_type] = {
                "z": z,
                "e": e,
                "n_active": n_active,
                "metrics": metrics,
                "name": name,
            }
            
            del model
        except Exception as e:
            print(f"  ❌ Error processing {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_model_data) == 0:
        raise RuntimeError("No models were successfully processed")

    if not args.metrics_only:
        # Run UMAP
        print("\n" + "=" * 60)
        print("UMAP")
        print("=" * 60)
        umap_data = {}
        for model_type, data in all_model_data.items():
            # FSQ/LFQ: use joint standardization and tuned UMAP params (they have dense discrete codes)
            use_fsq_lfq_tweaks = model_type in ("fsq", "lfq")
            z_emb, e_emb = run_umap(
                data["z"],
                data["e"],
                n_neighbors=25 if use_fsq_lfq_tweaks else args.umap_neighbors,
                min_dist=0.15 if use_fsq_lfq_tweaks else args.umap_min_dist,
                standardize=use_fsq_lfq_tweaks,
            )
            umap_data[model_type] = {**data, "z_emb": z_emb, "e_emb": e_emb}

        # Plot UMAP first (save before t-SNE in case it crashes)
        plot_all_models(
            umap_data, out_dir / "latent_vs_codes_all_umap.png", method="UMAP",
            show_metrics_in_title=False,
        )
        plot_all_models(
            umap_data, out_dir / "latent_vs_codes_main.png", method="UMAP",
            show_metrics_in_title=False,
        )
        if set(umap_data.keys()) == {"vq", "sim_vq", "lmb"}:
            plot_all_models(umap_data, out_dir / "latent_vs_codes_vq_simvq_lmb.png", method="UMAP")
            plot_all_models(
                umap_data, out_dir / "latent_vs_codes_vq_simvq_lmb_clean.png", method="UMAP",
                show_metrics_in_title=False, red_alpha_bump=0.1,
            )
            plot_all_models(
                umap_data, out_dir / "latent_vs_codes_vq_simvq_lmb_clean_no_subsample.png", method="UMAP",
                show_metrics_in_title=False, red_alpha_bump=0.1, subsample_red=False,
            )

        # Run t-SNE (with error handling in case of memory issues)
        if not args.skip_tsne:
            print("\n" + "=" * 60)
            print("t-SNE")
            print("=" * 60)
            tsne_data = {}
            try:
                for model_type, data in all_model_data.items():
                    print(f"  Running t-SNE for {model_type}...")
                    z_emb, e_emb = run_tsne(data["z"], data["e"], perplexity=args.tsne_perplexity, n_iter=args.tsne_n_iter)
                    tsne_data[model_type] = {**data, "z_emb": z_emb, "e_emb": e_emb}
                plot_all_models(tsne_data, out_dir / "latent_vs_codes_all_tsne.png", method="t-SNE")
                plot_all_models(tsne_data, out_dir / "latent_vs_codes_main_tsne.png", method="t-SNE")
            except Exception as e:
                print(f"  ⚠️  t-SNE failed (likely memory issue): {e}")
                print("  UMAP visualizations have been saved successfully.")
        else:
            print("\nSkipping t-SNE (--skip-tsne).")

    # Save metrics
    codebook_size = 16384
    metrics_name = "latent_vs_codes_all_metrics" + (f"_{args.output_suffix}" if args.output_suffix else "") + ".txt"
    metrics_path = out_dir / metrics_name
    with open(metrics_path, "w") as f:
        f.write("Latent vs active codes (main 16K models)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Images: {args.num_images}  |  Subsampled z: {list(all_model_data.values())[0]['z'].size(0)}\n\n")
        for model_type, data in all_model_data.items():
            util = 100 * data["n_active"] / codebook_size
            f.write(f"{data['name']}:\n")
            f.write(f"  Active codes: {data['n_active']} ({util:.1f}% utilization)\n")
            for k, v in data["metrics"].items():
                f.write(f"  {k}: {v:.4f}\n")
            f.write("\n")
    print(f"Saved: {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()

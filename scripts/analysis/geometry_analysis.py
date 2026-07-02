#!/usr/bin/env python3
"""
Codebook Geometry Visualisation for LGQ Paper.

Produces scientific figures proving that LGQ learns geometry in the way
Gersho theory predicts.

Figures:
  1. Voronoi cell visualisation (2D UMAP projection)
  2. Cell volume vs local data density (log-log)
  3. Assignment anisotropy analysis
  4. Gersho scaling law validation
  5. Convergence speed comparison (Phase 0 init vs baselines)

Usage:
    python geometry_analysis.py \
        --checkpoint results/lmb/lmb_ablation_cb8k/checkpoints/best_model.pt \
        --split val

    # Compare multiple models:
    python geometry_analysis.py \
        --checkpoint results/lmb/best_model.pt \
        --vq-checkpoint results/vq/best_model.pt \
        --fsq-checkpoint results/fsq/best_model.pt
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantization.model import ModelConfig, UnifiedAutoEncoder, QuantizerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Codebook Geometry Analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained LGQ (LMB) checkpoint")
    parser.add_argument("--vq-checkpoint", type=str, default=None,
                        help="Path to VQ checkpoint for comparison")
    parser.add_argument("--fsq-checkpoint", type=str, default=None,
                        help="Path to FSQ checkpoint for comparison")
    parser.add_argument("--data-root", type=str,
                        default="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet",
                        help="Path to ImageNet dataset")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "imagenet100"])
    parser.add_argument("--split", type=str, default="val",
                        choices=["val", "train"])
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of images to encode for analysis")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # For Figure 4 (scaling law): paths to checkpoints at different K values
    parser.add_argument("--scaling-checkpoints", type=str, nargs="*", default=None,
                        help="Checkpoint paths for scaling law (different K values)")
    parser.add_argument("--scaling-json", type=str, default=None,
                        help="JSON file with scaling results {K: {rfid, active_codes, mse}}")
    # For Figure 5 (convergence): paths to training logs
    parser.add_argument("--convergence-logs", type=str, nargs="*", default=None,
                        help="Paths to eval_metrics.csv for convergence comparison")
    parser.add_argument("--convergence-labels", type=str, nargs="*", default=None,
                        help="Labels for convergence comparison")
    # Phase 0 summary for d_eff reference
    parser.add_argument("--phase0-summary", type=str, default=None,
                        help="Path to phase0_results/summary.json")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint (same as phase0)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model_type = config.get("model_type", config.get("model", "lmb"))
    dim = config.get("dim", 128)
    embedding_dim = config.get("embedding_dim", 128)

    model_cfg = ModelConfig(base_ch=dim // 2, embedding_dim=embedding_dim)

    qconfig_kwargs = {}
    if model_type == "lmb":
        qconfig_kwargs = dict(
            num_bins=config.get("num_bins", 8192),
            lambda_peak=config.get("lambda_peak", 0.005),
            lambda_bins=config.get("lambda_bins", 0.005),
            lambda_floor=config.get("lambda_floor", 0.0),
            p_min=config.get("p_min", 0.0),
            tau=config.get("tau_end", 0.1),
            learn_widths=config.get("learn_widths", False),
            flatten_channels=config.get("flatten_channels", True),
            init_method=config.get("init_method", "random"),
            perchannel_fair=config.get("perchannel_fair", False),
            lmb_levels=config.get("lmb_levels", [16, 16, 8, 8]),
        )
    elif model_type == "vq":
        qconfig_kwargs = dict(
            codebook_size=config.get("codebook_size", 8192),
            commitment_weight=config.get("commitment_weight", 1.0),
            decay=config.get("decay", 0.8),
        )
    elif model_type == "fsq":
        qconfig_kwargs = dict(
            levels=config.get("levels", [8, 5, 5, 5]),
        )

    quantizer_config = QuantizerConfig(**qconfig_kwargs)
    model = UnifiedAutoEncoder(model_type, model_cfg, quantizer_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device).eval()
    return model, config, model_type


def get_imagenet100_synsets(data_root):
    train_dir = os.path.join(data_root, "train")
    all_synsets = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    return set(all_synsets[:100])


def create_dataset(data_root, dataset_name, image_size, split="val"):
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    split_path = os.path.join(data_root, split)
    full_dataset = datasets.ImageFolder(root=split_path, transform=transform)

    if dataset_name == "imagenet100":
        synsets = get_imagenet100_synsets(data_root)
        indices = [i for i, (path, _) in enumerate(full_dataset.samples)
                   if os.path.basename(os.path.dirname(path)) in synsets]
        full_dataset = Subset(full_dataset, indices)
    return full_dataset


@torch.no_grad()
def encode_and_assign(model, dataset, num_samples, batch_size, device, num_workers=4):
    """Encode images and get latent vectors + codeword assignments."""
    total = len(dataset)
    if num_samples < total:
        indices = torch.randperm(total)[:num_samples].tolist()
        subset = Subset(dataset, indices)
    else:
        subset = dataset

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_latents = []
    all_indices = []

    for images, _ in loader:
        images = images.to(device)
        z = model.enc(images)
        B, C, H, W = z.shape

        # Get normalized latents (same pipeline as encode)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat)

        if model.pre_quant is not None:
            z_2d = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_proj = model.pre_quant(z_2d)
            C_out = z_proj.shape[1]
            z_norm = z_proj.permute(0, 2, 3, 1).contiguous().view(B * H * W, C_out)

        all_latents.append(z_norm.cpu())

        # Get indices
        idx = model.encode(images)
        if idx.dim() == 3:
            idx = idx.view(-1)  # flatten per-channel
        else:
            idx = idx.view(-1)
        all_indices.append(idx.cpu())

    Z = torch.cat(all_latents, dim=0)
    indices = torch.cat(all_indices, dim=0)

    # Subsample if too many patches
    total_patches = Z.shape[0]
    if total_patches > num_samples * 16:
        sel = torch.randperm(total_patches)[:num_samples * 8]
        Z = Z[sel]
        indices = indices[sel]

    return Z.numpy(), indices.numpy()


def get_codebook(model, model_type):
    """Extract codebook vectors from model."""
    if model_type == "lmb":
        if hasattr(model.quantize, "centers"):
            return model.quantize.centers.detach().cpu().numpy()
    elif model_type == "vq":
        if hasattr(model.quantize, "_codebook"):
            cb = model.quantize._codebook
            if hasattr(cb, "embed"):
                return cb.embed.squeeze(0).detach().cpu().numpy()
        if hasattr(model.quantize, "codebook"):
            return model.quantize.codebook.detach().cpu().numpy()
    return None


# =============================================================================
# FIGURE 1: Voronoi cell visualisation (2D UMAP)
# =============================================================================
def figure1_voronoi_umap(models_data, output_dir):
    """
    Voronoi cell visualisation in 2D UMAP space.
    models_data: list of (name, Z, assignments, codebook)
    """
    print("\nFigure 1: Voronoi UMAP visualisation")
    try:
        import umap
    except ImportError:
        print("  WARNING: umap-learn not installed, skipping Figure 1")
        return

    n_models = len(models_data)
    fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    for ax, (name, Z, assignments, codebook) in zip(axes, models_data):
        # Subsample for UMAP speed
        max_pts = 8000
        if len(Z) > max_pts:
            sel = np.random.choice(len(Z), max_pts, replace=False)
            Z_sub = Z[sel]
            assign_sub = assignments[sel]
        else:
            Z_sub = Z
            assign_sub = assignments

        # Fit UMAP on latents + codebook together
        if codebook is not None and len(codebook) > 0:
            combined = np.vstack([Z_sub, codebook])
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            embedding = reducer.fit_transform(combined)
            Z_2d = embedding[:len(Z_sub)]
            cb_2d = embedding[len(Z_sub):]
        else:
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
            Z_2d = reducer.fit_transform(Z_sub)
            cb_2d = None

        # Colour by assignment
        unique_codes = np.unique(assign_sub)
        n_colors = min(len(unique_codes), 50)
        cmap = cm.get_cmap("tab20", n_colors)

        # Map codes to colour indices
        code_to_color = {c: i % n_colors for i, c in enumerate(unique_codes)}
        colors = [cmap(code_to_color[a]) for a in assign_sub]

        ax.scatter(Z_2d[:, 0], Z_2d[:, 1], c=colors, s=1, alpha=0.3, rasterized=True)
        if cb_2d is not None:
            ax.scatter(cb_2d[:, 0], cb_2d[:, 1], c="red", s=30, marker="x",
                       linewidths=1, zorder=5, label="Codewords")
        ax.set_title(name, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Voronoi Cells in UMAP Space", fontsize=16, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig1_voronoi_umap.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 2: Cell volume vs local data density
# =============================================================================
def figure2_volume_vs_density(models_data, d_eff, output_dir):
    """Log-log scatter: cell volume vs local data density with Gersho reference."""
    print("\nFigure 2: Cell volume vs local density")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    markers = ["o", "s", "^"]
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (name, Z, assignments, codebook) in enumerate(models_data):
        if codebook is None or len(codebook) == 0:
            continue

        K = codebook.shape[0]
        counts = np.bincount(assignments.astype(int), minlength=K).astype(float)
        active_mask = counts > 5  # need enough points for density estimate

        # Cell volume ~ 1/count (normalised)
        total_count = counts.sum()
        cell_volume = np.zeros(K)
        cell_volume[active_mask] = total_count / (counts[active_mask] * K)

        # Local density via Gaussian KDE (Scott's rule)
        from scipy.spatial.distance import cdist
        C_dim = Z.shape[1]
        h = len(Z) ** (-1.0 / (C_dim + 4)) * Z.std()  # Scott's rule
        h = max(h, 1e-6)

        # Compute density at each codeword (batch for memory)
        density = np.zeros(K)
        batch = 2000
        for i in range(0, len(Z), batch):
            z_batch = Z[i:i+batch]
            d = cdist(codebook[active_mask], z_batch, metric="euclidean")
            density[active_mask] += np.exp(-0.5 * (d / h) ** 2).sum(axis=1)
        density[active_mask] /= (len(Z) * (h ** C_dim) * (2 * np.pi) ** (C_dim / 2))

        # Filter for plotting
        valid = active_mask & (density > 0) & (cell_volume > 0)
        if valid.sum() < 10:
            print(f"  {name}: too few valid codewords ({valid.sum()}), skipping")
            continue

        log_vol = np.log10(cell_volume[valid])
        log_dens = np.log10(density[valid])

        ax.scatter(log_dens, log_vol, s=15, alpha=0.5, label=name,
                   marker=markers[idx % 3], color=colors_list[idx % 3])

        # Fit power law
        coeffs = np.polyfit(log_dens, log_vol, 1)
        x_fit = np.linspace(log_dens.min(), log_dens.max(), 100)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--",
                color=colors_list[idx % 3], alpha=0.8,
                label=f"{name} fit (slope={coeffs[0]:.2f})")

    # Gersho theoretical slope
    if d_eff is not None and d_eff > 0:
        gersho_slope = -(d_eff + 2) / d_eff
        ax.plot([], [], "k--", label=f"Gersho theory (slope={gersho_slope:.2f})")
        # Draw reference line
        x_ref = np.array([ax.get_xlim()[0], ax.get_xlim()[1]])
        y_ref = gersho_slope * (x_ref - x_ref.mean()) + np.mean(ax.get_ylim())
        ax.plot(x_ref, y_ref, "k--", alpha=0.4, linewidth=2)

    ax.set_xlabel("log₁₀(Local Density)", fontsize=12)
    ax.set_ylabel("log₁₀(Cell Volume)", fontsize=12)
    ax.set_title("Cell Volume vs Local Data Density", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "fig2_volume_vs_density.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 3: Assignment anisotropy analysis
# =============================================================================
def figure3_anisotropy(models_data, output_dir):
    """Distribution of anisotropy ratios per Voronoi cell."""
    print("\nFigure 3: Assignment anisotropy")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, (name, Z, assignments, codebook) in enumerate(models_data):
        if codebook is None:
            continue

        K = codebook.shape[0]
        counts = np.bincount(assignments.astype(int), minlength=K)
        anisotropy_ratios = []
        cell_volumes = []

        for k in range(K):
            mask = assignments == k
            n_assigned = int(counts[k])
            if n_assigned < Z.shape[1] + 2:  # need enough points for covariance
                continue

            Z_cell = Z[mask]
            cov = np.cov(Z_cell, rowvar=False)
            try:
                eigvals = np.linalg.eigvalsh(cov)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) >= 2:
                    ratio = eigvals[-1] / eigvals[0]
                    anisotropy_ratios.append(ratio)
                    cell_volumes.append(1.0 / n_assigned)
            except np.linalg.LinAlgError:
                continue

        if len(anisotropy_ratios) == 0:
            print(f"  {name}: no valid cells for anisotropy")
            continue

        anisotropy_ratios = np.array(anisotropy_ratios)
        cell_volumes = np.array(cell_volumes)

        # Histogram of anisotropy ratios
        log_ratios = np.log10(anisotropy_ratios.clip(min=1))
        ax1.hist(log_ratios, bins=30, alpha=0.5, label=f"{name} (median={np.median(anisotropy_ratios):.1f})",
                 color=colors_list[idx % 3], density=True)

        # Anisotropy vs cell volume
        ax2.scatter(np.log10(cell_volumes), np.log10(anisotropy_ratios.clip(min=1)),
                    s=10, alpha=0.4, label=name, color=colors_list[idx % 3])

    ax1.set_xlabel("log₁₀(Anisotropy Ratio)", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.set_title("Distribution of Cell Anisotropy", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("log₁₀(Cell Volume)", fontsize=12)
    ax2.set_ylabel("log₁₀(Anisotropy Ratio)", fontsize=12)
    ax2.set_title("Anisotropy vs Cell Volume", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig3_anisotropy.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 4: Gersho scaling law validation
# =============================================================================
def figure4_scaling_law(args, d_eff, output_dir):
    """
    log(distortion) vs log(active_codes) — should have slope -2/d_eff.
    """
    print("\nFigure 4: Gersho scaling law")

    scaling_data = None
    if args.scaling_json and os.path.exists(args.scaling_json):
        with open(args.scaling_json) as f:
            scaling_data = json.load(f)
    elif args.scaling_checkpoints:
        # Load metrics from each checkpoint's eval log
        scaling_data = {}
        for cp_path in args.scaling_checkpoints:
            cp_dir = Path(cp_path).parent.parent
            config_path = cp_dir / "config.json"
            eval_csv = cp_dir / "eval_metrics.csv"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                K = cfg.get("num_bins", cfg.get("codebook_size", 0))
            else:
                K = 0
            if eval_csv.exists():
                import csv
                with open(eval_csv) as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                if rows:
                    last = rows[-1]
                    scaling_data[str(K)] = {
                        "rfid": float(last.get("val_rfid", 0)),
                        "active_codes": int(float(last.get("val_active_codes", 0))),
                        "mse": float(last.get("val_rec_loss", 0)),
                    }

    if not scaling_data:
        print("  No scaling data provided, skipping Figure 4")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    Ks = sorted([int(k) for k in scaling_data.keys()])
    active_codes = [scaling_data[str(k)]["active_codes"] for k in Ks]
    mses = [scaling_data[str(k)]["mse"] for k in Ks]
    rfids = [scaling_data[str(k)]["rfid"] for k in Ks]

    log_active = np.log2(np.array(active_codes).clip(min=1))
    log_mse = np.log2(np.array(mses).clip(min=1e-10))

    # Plot distortion vs active codes
    ax1.plot(log_active, log_mse, "bo-", markersize=8, label="LGQ")
    for i, k in enumerate(Ks):
        ax1.annotate(f"K={k}", (log_active[i], log_mse[i]),
                     textcoords="offset points", xytext=(5, 5), fontsize=8)

    # Fit line
    if len(Ks) >= 3:
        coeffs = np.polyfit(log_active, log_mse, 1)
        x_fit = np.linspace(log_active.min(), log_active.max(), 100)
        ax1.plot(x_fit, np.polyval(coeffs, x_fit), "b--", alpha=0.5,
                 label=f"Fit (slope={coeffs[0]:.3f})")

        # Gersho prediction: slope = -2/d_eff
        if d_eff is not None and d_eff > 0:
            gersho_slope = -2.0 / d_eff
            y_ref = gersho_slope * (x_fit - x_fit.mean()) + log_mse.mean()
            ax1.plot(x_fit, y_ref, "r--", alpha=0.7,
                     label=f"Gersho (-2/d_eff={gersho_slope:.3f})")

    ax1.set_xlabel("log₂(Active Codes)", fontsize=12)
    ax1.set_ylabel("log₂(Distortion)", fontsize=12)
    ax1.set_title("Gersho Scaling Law", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # rFID vs K
    ax2.semilogx(Ks, rfids, "go-", markersize=8, base=2)
    ax2.set_xlabel("Codebook Size K", fontsize=12)
    ax2.set_ylabel("rFID", fontsize=12)
    ax2.set_title("Reconstruction FID vs Codebook Size", fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Mark N* if available
    if args.phase0_summary and os.path.exists(args.phase0_summary):
        with open(args.phase0_summary) as f:
            p0 = json.load(f)
        N_star = p0.get("N_star", None)
        if N_star:
            ax2.axvline(N_star, color="red", linestyle=":", alpha=0.7, label=f"N*={N_star}")
            ax2.legend(fontsize=9)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig4_scaling_law.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# FIGURE 5: Convergence speed comparison
# =============================================================================
def figure5_convergence(args, output_dir):
    """Plot rFID and active codes vs epoch for different inits."""
    print("\nFigure 5: Convergence comparison")

    if not args.convergence_logs:
        print("  No convergence logs provided, skipping Figure 5")
        return

    import csv

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors_list = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    labels = args.convergence_labels or [f"Run {i}" for i in range(len(args.convergence_logs))]

    for idx, (log_path, label) in enumerate(zip(args.convergence_logs, labels)):
        if not os.path.exists(log_path):
            print(f"  WARNING: {log_path} not found, skipping")
            continue

        with open(log_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        epochs = [int(float(r.get("epoch", i))) for i, r in enumerate(rows)]
        rfids = [float(r.get("val_rfid", 0)) for r in rows]
        active = [int(float(r.get("val_active_codes", 0))) for r in rows]

        color = colors_list[idx % len(colors_list)]
        ax1.plot(epochs, rfids, "-o", markersize=3, color=color, label=label)
        ax2.plot(epochs, active, "-o", markersize=3, color=color, label=label)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("rFID", fontsize=12)
    ax1.set_title("rFID vs Training Epoch", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Active Codes", fontsize=12)
    ax2.set_title("Active Codes vs Training Epoch", fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(output_dir, "fig5_convergence.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load d_eff from Phase 0 if available
    d_eff = None
    if args.phase0_summary and os.path.exists(args.phase0_summary):
        with open(args.phase0_summary) as f:
            p0 = json.load(f)
        d_eff = p0.get("d_eff", None)
        print(f"Phase 0 d_eff = {d_eff}")

    # Load primary model (LGQ/LMB)
    print(f"\nLoading primary model: {args.checkpoint}")
    model, config, model_type = load_model_from_checkpoint(args.checkpoint, device)

    # Create dataset
    dataset = create_dataset(args.data_root, args.dataset, args.image_size, args.split)
    print(f"Dataset: {len(dataset)} images ({args.split})")

    # Encode and get assignments for primary model
    print(f"\nEncoding with {model_type}...")
    Z, assignments = encode_and_assign(model, dataset, args.num_samples,
                                        args.batch_size, device, args.num_workers)
    codebook = get_codebook(model, model_type)

    # If d_eff not from Phase 0, compute it
    if d_eff is None:
        Z_t = torch.from_numpy(Z).float()
        Z_centered = Z_t - Z_t.mean(dim=0, keepdim=True)
        if Z.shape[1] > 256:
            _, S, _ = torch.linalg.svd(Z_centered, full_matrices=False)
            eigs = (S ** 2) / len(Z)
        else:
            Sigma = (Z_centered.T @ Z_centered) / len(Z)
            eigs = torch.linalg.eigvalsh(Sigma).flip(0)
        eigs = eigs.clamp(min=0)
        d_eff = float((eigs.sum() ** 2) / (eigs ** 2).sum())
        print(f"Computed d_eff = {d_eff:.2f}")

    models_data = [(model_type.upper(), Z, assignments, codebook)]

    # Load comparison models if provided
    for ckpt_path, label in [(args.vq_checkpoint, "VQ"), (args.fsq_checkpoint, "FSQ")]:
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"\nLoading comparison model: {label} from {ckpt_path}")
            m, _, mt = load_model_from_checkpoint(ckpt_path, device)
            Z_cmp, assign_cmp = encode_and_assign(m, dataset, args.num_samples,
                                                   args.batch_size, device, args.num_workers)
            cb_cmp = get_codebook(m, mt)
            models_data.append((label, Z_cmp, assign_cmp, cb_cmp))
            del m
            torch.cuda.empty_cache()

    # Generate figures
    figure1_voronoi_umap(models_data, output_dir)
    figure2_volume_vs_density(models_data, d_eff, output_dir)
    figure3_anisotropy(models_data, output_dir)
    figure4_scaling_law(args, d_eff, output_dir)
    figure5_convergence(args, output_dir)

    print(f"\n{'='*60}")
    print(f"All figures saved to {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

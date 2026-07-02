#!/usr/bin/env python3
"""
Plot "PROOF: Bins Are Learnable" — improved visualization with t-SNE/UMAP 2D projection.

Creates a 4-panel figure:
  1. Distribution: Initial vs Final (histogram)
  2. Bin Movement (bar chart, sorted by movement magnitude)
  3. 2D Projection: Initial vs Final (t-SNE or UMAP with arrows)
  4. Text summary (stats, evidence of learning)

Usage:
  python scripts/plot_bins_learnable_proof.py --exp-dir results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd
  python scripts/plot_bins_learnable_proof.py --exp-dir results/lmb/lmb_fixed_init --method umap
  python scripts/plot_bins_learnable_proof.py --exp-dir results/lmb/lmb_fixed_init --method both

When the first checkpoint is missing: the script uses the earliest available epoch as "initial"
and the latest as "final". If only best_model.pt and latest_model.pt exist (no checkpoint_epoch_*.pt),
it loads those two so you still get "Best vs Last epoch" and the individual-centers-over-epochs
panel shows the trajectory between those two snapshots.

To get true first→last (epoch 1 vs last): train LMB with --save-initial-centers; the script will
then use analysis_epoch001_centers.pt as "initial" when present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_centers_from_checkpoint(checkpoint_path: Path) -> np.ndarray | None:
    """Load bin centers from a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = checkpoint.get("model_state_dict", {})
        for key, value in model_state.items():
            if "quantize.centers" in key:
                return value.cpu().numpy()
        return None
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def _load_centers(centers: np.ndarray, flatten_channels: bool) -> np.ndarray:
    """Reshape centers to [K, C] (each row = codebook vector)."""
    if flatten_channels:
        return centers  # [K, C]
    return centers.T  # [C, K] -> [K, C]


def get_initial_final_centers(exp_dir: Path) -> tuple[int, np.ndarray, int, np.ndarray] | None:
    """Load initial and final centers: earliest available epoch → latest. Returns (initial_epoch, initial, final_epoch, final).
    When only best_model.pt and latest_model.pt exist, uses earlier of the two as 'initial' and later as 'final'."""
    result = get_all_epoch_centers(exp_dir)
    if result is None or len(result) < 2:
        return None
    epochs = sorted(result.keys())
    initial_epoch = epochs[0]
    final_epoch = epochs[-1]
    # Prefer epoch 1 as initial when available (full trajectory)
    if 1 in result:
        initial_epoch = 1
    c1 = result[initial_epoch]
    c2 = result[final_epoch]
    n = min(len(c1), len(c2))
    return initial_epoch, c1[:n], final_epoch, c2[:n]


def get_all_epoch_centers(exp_dir: Path, max_epochs: int | None = None) -> dict[int, np.ndarray] | None:
    """Load centers from all checkpoints. Returns {epoch: centers[K,C]}. max_epochs=None loads all.
    Uses checkpoint_epoch_*.pt when present; if fewer than 2, also loads latest_model.pt and best_model.pt
    so you can plot 'best + last epoch' when the first checkpoint is missing."""
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    config_path = exp_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    flatten_channels = config.get("flatten_channels", False)

    epoch_centers: dict[int, np.ndarray] = {}

    # 1) Load per-epoch checkpoints if present
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if max_epochs is not None and len(checkpoints) > max_epochs:
        indices = [0] + list(np.linspace(1, len(checkpoints) - 2, max(0, max_epochs - 2), dtype=int)) + [len(checkpoints) - 1]
        indices = sorted(set(indices))
        checkpoints = [checkpoints[i] for i in indices]
    for cp_path in checkpoints:
        try:
            epoch = int(cp_path.stem.split("_")[-1])
            centers = load_centers_from_checkpoint(cp_path)
            if centers is not None:
                epoch_centers[epoch] = _load_centers(centers, flatten_channels)
        except (ValueError, IndexError):
            continue

    # 2) If we don't have epoch 1 but analysis_epoch001_centers.pt exists (from --save-initial-centers), use it
    analysis_path = checkpoint_dir / "analysis_epoch001_centers.pt"
    if 1 not in epoch_centers and analysis_path.exists():
        try:
            data = torch.load(analysis_path, map_location="cpu")
            c = data.get("centers")
            if c is not None:
                c = c.numpy() if hasattr(c, "numpy") else c
                flat = data.get("flatten_channels", flatten_channels)
                epoch_centers[1] = _load_centers(c, flat)
        except Exception:
            pass

    # 3) If we have fewer than 2 snapshots, add latest_model.pt and best_model.pt (epoch from checkpoint)
    if len(epoch_centers) < 2:
        for name in ("latest_model.pt", "best_model.pt"):
            cp_path = checkpoint_dir / name
            if not cp_path.exists():
                continue
            try:
                ckpt = torch.load(cp_path, map_location="cpu")
                epoch = int(ckpt.get("epoch", 0))
                centers = load_centers_from_checkpoint(cp_path)
                if centers is not None:
                    epoch_centers[epoch] = _load_centers(centers, flatten_channels)
            except (ValueError, TypeError, KeyError):
                continue

    return epoch_centers if len(epoch_centers) >= 2 else None


def load_eval_metrics(exp_dir: Path) -> dict[int, dict[str, float]] | None:
    """Load eval_metrics.csv. Returns {epoch: {metric: value}} so we can plot from Epoch 1 (even without checkpoint)."""
    csv_path = exp_dir / "eval_metrics.csv"
    if not csv_path.exists():
        return None
    out: dict[int, dict[str, float]] = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row.get("epoch", 0))
            except (ValueError, TypeError):
                continue
            metrics = {}
            for key in ("val_active_codes", "val_rec_loss", "val_perplexity", "val_psnr", "val_rfid"):
                v = row.get(key)
                if v is None or str(v).strip() == "":
                    continue
                try:
                    metrics[key] = float(v)
                except ValueError:
                    continue
            if metrics:
                # Keep last row per epoch if duplicates
                out[epoch] = metrics
    return out if out else None


def run_tsne(X: np.ndarray, perplexity: float = 30, max_points: int | None = None) -> np.ndarray:
    """Run t-SNE on X. Samples if too large (max_points applies to total rows)."""
    from sklearn.manifold import TSNE

    if max_points is not None and len(X) > max_points:
        idx = np.linspace(0, len(X) - 1, max_points, dtype=int)
        X = X[idx]
    perp = min(perplexity, max(2, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000, verbose=0)
    return tsne.fit_transform(X)


def run_umap(X: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1, max_points: int | None = None) -> np.ndarray:
    """Run UMAP on X. Samples if too large."""
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required. Install with: pip install umap-learn")

    if max_points is not None and len(X) > max_points:
        idx = np.linspace(0, len(X) - 1, max_points, dtype=int)
        X = X[idx]
    n_n = min(n_neighbors, max(2, len(X) - 1))
    reducer = umap.UMAP(
        n_neighbors=n_n,
        min_dist=min_dist,
        n_components=2,
        metric="euclidean",
        random_state=42,
        verbose=False,
    )
    return reducer.fit_transform(X)


def plot_proof(
    exp_dir: Path,
    output_path: Path,
    initial_epoch: int,
    initial_centers: np.ndarray,
    final_epoch: int,
    final_centers: np.ndarray,
    method: str = "tsne",
    max_points: int = 2000,
    perplexity: float = 30,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    epoch_centers: dict[int, np.ndarray] | None = None,
    n_individual_centers: int = 20,
    eval_metrics: dict[int, dict[str, float]] | None = None,
) -> None:
    """Create the PROOF figure (2x2 layout).

    Layout:
      Row 0: (a) 2D Projection  | (b) Distribution
      Row 1: (c) Sample of N centers over epochs | (d) Bin movement over time
    """

    # Flatten for distribution
    initial_flat = initial_centers.flatten()
    final_flat = final_centers.flatten()
    num_bins = len(initial_centers)

    # Movement per bin (L2 distance initial → final)
    movement = np.linalg.norm(final_centers - initial_centers, axis=1)
    mean_movement = float(np.mean(movement))
    max_movement = float(np.max(movement))
    init_std = float(initial_flat.std())
    fin_std = float(final_flat.std())

    # Sample for 2D projection
    n_plot = min(max_points, num_bins)
    if num_bins > max_points:
        idx = np.linspace(0, num_bins - 1, n_plot, dtype=int)
        init_sub = initial_centers[idx]
        fin_sub = final_centers[idx]
    else:
        init_sub = initial_centers
        fin_sub = final_centers

    # 2D embedding
    all_data = np.vstack([init_sub, fin_sub])
    print(f"  Computing {method.upper()} on {len(all_data)} points...")
    if method.lower() == "umap":
        emb = run_umap(all_data, n_neighbors=n_neighbors, min_dist=min_dist)
    else:
        emb = run_tsne(all_data, perplexity=perplexity)
    init_emb = emb[:n_plot]
    fin_emb = emb[n_plot:]

    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)

    def panel_label(ax: plt.Axes, label: str) -> None:
        ax.text(-0.08, 1.02, label, transform=ax.transAxes, fontsize=12, fontweight="bold", va="bottom", ha="right")

    # Row 0, Col 0: 2D Projection
    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "(a)")
    num_lines = min(400, n_plot)
    line_idx = np.linspace(0, n_plot - 1, num_lines, dtype=int)
    for i in line_idx:
        ax1.plot([init_emb[i, 0], fin_emb[i, 0]], [init_emb[i, 1], fin_emb[i, 1]], "C1", alpha=0.12, linewidth=0.6, zorder=1)
    ax1.scatter(init_emb[:, 0], init_emb[:, 1], c="blue", marker="o", s=15, alpha=0.5, label=f"Epoch {initial_epoch}", zorder=2)
    ax1.scatter(fin_emb[:, 0], fin_emb[:, 1], c="red", marker="x", s=15, alpha=0.5, label=f"Epoch {final_epoch}", zorder=3)
    ax1.set_xlabel(f"{method.upper()} 1")
    ax1.set_ylabel(f"{method.upper()} 2")
    ax1.set_title(f"2D Projection: Epoch {initial_epoch} vs Epoch {final_epoch} ({method.upper()})")
    ax1.legend(loc="best", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal", adjustable="box")

    # Row 0, Col 1: Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    panel_label(ax2, "(b)")
    bins_hist = np.linspace(min(initial_flat.min(), final_flat.min()), max(initial_flat.max(), final_flat.max()), 60)
    ax2.hist(initial_flat, bins=bins_hist, alpha=0.5, label=f"Epoch {initial_epoch}", color="blue", density=True)
    ax2.hist(final_flat, bins=bins_hist, alpha=0.5, label=f"Epoch {final_epoch}", color="orangered", density=True)
    ax2.set_xlabel("Bin Center Value")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Distribution: Epoch {initial_epoch} vs Epoch {final_epoch}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.text(0.02, 0.98, f"σ(Epoch {initial_epoch}): {init_std:.4f}\nσ(Epoch {final_epoch}): {fin_std:.4f}", transform=ax2.transAxes, fontsize=9, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.8", alpha=0.9))

    # Row 1, Col 0: Sample of N individual centers over epochs
    ax3 = fig.add_subplot(gs[1, 0])
    panel_label(ax3, "(c)")
    if epoch_centers and len(epoch_centers) >= 2:
        epochs_sorted = sorted(epoch_centers.keys())
        # Pick N bins uniformly across the codebook
        bin_indices = np.linspace(0, num_bins - 1, min(n_individual_centers, num_bins), dtype=int)
        for bi, bin_idx in enumerate(bin_indices):
            values = [epoch_centers[e][bin_idx].mean() for e in epochs_sorted]
            ax3.plot(epochs_sorted, values, "o-", linewidth=1, markersize=3, alpha=0.7)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Center Value (mean over dims)")
        ax3.set_title(f"Sample of {len(bin_indices)} Individual Centers Over Epochs")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Need multiple epoch checkpoints\nfor trajectory plot", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("Sample of Individual Centers Over Epochs")

    # Row 1, Col 1: Bin Movement Over Time
    ax4 = fig.add_subplot(gs[1, 1])
    panel_label(ax4, "(d)")
    if epoch_centers and len(epoch_centers) >= 2:
        epochs_sorted = sorted(epoch_centers.keys())
        mean_movements = []
        epoch_transitions = []
        for i in range(len(epochs_sorted) - 1):
            e1, e2 = epochs_sorted[i], epochs_sorted[i + 1]
            c1, c2 = epoch_centers[e1], epoch_centers[e2]
            n = min(len(c1), len(c2))
            diff = np.abs(c2[:n] - c1[:n])
            mean_movements.append(float(np.mean(diff)))
            epoch_transitions.append(e2)
        ax4.plot(epoch_transitions, mean_movements, "go-", linewidth=2, markersize=6, alpha=0.8)
        ax4.set_xlabel("Epoch (transition)")
        ax4.set_ylabel("Mean Absolute Movement")
        ax4.set_title("Bin Movement Over Time")
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Need multiple epoch checkpoints\nfor movement-over-time plot", ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Bin Movement Over Time")

    fig.suptitle("PROOF: Bins Are Learnable", fontsize=14, fontweight="bold", y=1.01)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PROOF: Bins Are Learnable (with t-SNE/UMAP)")
    parser.add_argument("--exp-dir", type=str, default="results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd", help="LMB experiment directory")
    parser.add_argument("--output-dir", type=str, default="plots", help="Output directory")
    parser.add_argument("--method", type=str, choices=["tsne", "umap", "both"], default="tsne", help="2D projection method")
    parser.add_argument("--max-points", type=int, default=2000, help="Max points for t-SNE/UMAP")
    parser.add_argument("--perplexity", type=float, default=30, help="t-SNE perplexity")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    parser.add_argument("--n-individual-centers", type=int, default=20, help="Number of individual bin trajectories")
    parser.add_argument("--output-suffix", type=str, default="", help="Suffix for output filenames (e.g. lmb_fair)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        print(f"Error: Experiment directory not found: {exp_dir}")
        sys.exit(1)

    result = get_initial_final_centers(exp_dir)
    if result is None:
        print("Could not load initial and final centers.")
        sys.exit(1)

    initial_epoch, initial_centers, final_epoch, final_centers = result
    epoch_centers_all = get_all_epoch_centers(exp_dir, max_epochs=30)
    eval_metrics = load_eval_metrics(exp_dir)
    if epoch_centers_all:
        print(f"Loaded: Epoch {initial_epoch} → {final_epoch}, shape {initial_centers.shape}, {len(epoch_centers_all)} epochs for trajectories")
    else:
        print(f"Loaded: Epoch {initial_epoch} → {final_epoch}, shape {initial_centers.shape} (no trajectory data)")
    if eval_metrics:
        print(f"Metrics: {len(eval_metrics)} epochs from eval_metrics.csv (Epoch 1 included: {1 in eval_metrics})")

    out_dir = Path(args.output_dir)
    n_indiv = args.n_individual_centers

    suffix = f"_{args.output_suffix}" if args.output_suffix else ""
    for m in (["tsne", "umap"] if args.method == "both" else [args.method]):
        out_path = out_dir / f"bins_learnable_proof{suffix}_{m}.png"
        plot_proof(
            exp_dir,
            out_path,
            initial_epoch,
            initial_centers,
            final_epoch,
            final_centers,
            method=m,
            max_points=args.max_points,
            perplexity=args.perplexity,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            epoch_centers=epoch_centers_all,
            n_individual_centers=n_indiv,
            eval_metrics=eval_metrics,
        )

    # Also save bins_learnable_proof.png as the default
    import shutil

    default_path = out_dir / f"bins_learnable_proof{suffix}.png" if suffix else out_dir / "bins_learnable_proof.png"
    src = out_dir / f"bins_learnable_proof{suffix}_{args.method if args.method != 'both' else 'tsne'}.png"
    if src.exists():
        shutil.copy(src, default_path)
        print(f"Done. Default saved as {default_path}")


if __name__ == "__main__":
    main()

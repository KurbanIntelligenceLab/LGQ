#!/usr/bin/env python3
"""
Plot center drift for LGQ (single codebook): Δ_k = ‖v_k(T) - v_k(0)‖_2.

Modes:
  --plot drift:     (a) Endpoint drift Δ_k per entry (sorted), (b) mean drift per epoch.
  --plot start-end: 2D view of sample codebook entries: start → end with arrows.
  --plot evolution: Figure 6 style — left: sample trajectories over epochs; right: distribution initial vs final.
                    Default --exp-dir for start-end/evolution: results/lmb/lmb_ablation_reg_strong.

Loads from checkpoints/checkpoint_epoch_*.pt or checkpoints/ (or exp dir) best_model.pt + latest_model.pt.
Use --demo when no checkpoints exist to generate synthetic figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_centers_from_checkpoint(checkpoint_path: Path) -> np.ndarray | None:
    """Load codebook centers from checkpoint. Raw shape: LMB [C,K] or [K,C], VQ [K,D]."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_state = checkpoint.get("model_state_dict", {})
        for key, value in model_state.items():
            if "quantize.centers" in key:
                return value.cpu().numpy()
            if "quantize._codebook.embed" in key:
                c = value.cpu().numpy()
                if c.ndim == 3:
                    c = c.squeeze(0)
                return c
        return None
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None


def centers_to_codebook_vectors(centers: np.ndarray, flatten_channels: bool, model: str) -> np.ndarray:
    """Return [K, dim]: each row is one codebook vector."""
    if model == "lmb":
        if flatten_channels:
            return centers  # [K, C]
        return centers.T  # [C, K] -> [K, C]
    # VQ / SimVQ: already [K, D]
    return centers


def _load_config(exp_dir: Path) -> tuple[str, bool]:
    config_path = exp_dir / "config.json"
    model, flatten_channels = "lmb", False
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        model = config.get("model", "lmb")
        flatten_channels = config.get("flatten_channels", False)
    return model, flatten_channels


def _centers_from_ckpt(cp_path: Path, model: str, flatten_channels: bool) -> np.ndarray | None:
    raw = load_centers_from_checkpoint(cp_path)
    if raw is None:
        return None
    return centers_to_codebook_vectors(raw, flatten_channels, model)


def get_initial_final_centers(exp_dir: Path) -> tuple[int, np.ndarray, int, np.ndarray] | None:
    """Load initial and final codebook centers. Tries checkpoints/, then best/latest in checkpoints/, then in exp_dir.
    Returns (initial_epoch, initial [K,C], final_epoch, final [K,C]) or None."""
    model, flatten_channels = _load_config(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    epoch_centers: dict[int, np.ndarray] = {}

    # 1) Per-epoch checkpoints
    if ckpt_dir.exists():
        for cp_path in sorted(ckpt_dir.glob("checkpoint_epoch_*.pt")):
            try:
                epoch = int(cp_path.stem.split("_")[-1])
                c = _centers_from_ckpt(cp_path, model, flatten_channels)
                if c is not None:
                    epoch_centers[epoch] = c
            except (ValueError, IndexError):
                continue

    # 2) If fewer than 2 snapshots, try best_model.pt and latest_model.pt (in checkpoints/ or exp_dir)
    if len(epoch_centers) < 2:
        for base in ([ckpt_dir] if ckpt_dir.exists() else []) + [exp_dir]:
            for name in ("latest_model.pt", "best_model.pt"):
                cp_path = base / name
                if not cp_path.exists():
                    continue
                try:
                    ckpt = torch.load(cp_path, map_location="cpu", weights_only=False)
                    epoch = int(ckpt.get("epoch", 0))
                    c = _centers_from_ckpt(cp_path, model, flatten_channels)
                    if c is not None:
                        epoch_centers[epoch] = c
                except (ValueError, TypeError, KeyError):
                    continue

    if len(epoch_centers) < 2:
        return None
    epochs_sorted = sorted(epoch_centers.keys())
    initial_epoch = epochs_sorted[0]
    if 1 in epoch_centers:
        initial_epoch = 1
    final_epoch = epochs_sorted[-1]
    return initial_epoch, epoch_centers[initial_epoch], final_epoch, epoch_centers[final_epoch]


def get_epoch_centers_any(exp_dir: Path, max_epochs: int | None = None) -> dict[int, np.ndarray] | None:
    """Load all available epoch centers: checkpoint_epoch_*.pt and/or best/latest. Returns {epoch: centers [K, dim]}."""
    model, flatten_channels = _load_config(exp_dir)
    ckpt_dir = exp_dir / "checkpoints"
    epoch_centers: dict[int, np.ndarray] = {}

    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
        if max_epochs is not None and len(checkpoints) > max_epochs:
            indices = [0] + list(np.linspace(1, len(checkpoints) - 2, max(0, max_epochs - 2), dtype=int)) + [len(checkpoints) - 1]
            indices = sorted(set(indices))
            checkpoints = [checkpoints[i] for i in indices]
        for cp_path in checkpoints:
            try:
                epoch = int(cp_path.stem.split("_")[-1])
                c = _centers_from_ckpt(cp_path, model, flatten_channels)
                if c is not None:
                    epoch_centers[epoch] = c
            except (ValueError, IndexError):
                continue

    if len(epoch_centers) < 2:
        for base in ([ckpt_dir] if ckpt_dir.exists() else []) + [exp_dir]:
            for name in ("latest_model.pt", "best_model.pt"):
                cp_path = base / name
                if not cp_path.exists():
                    continue
                try:
                    ckpt = torch.load(cp_path, map_location="cpu", weights_only=False)
                    epoch = int(ckpt.get("epoch", 0))
                    c = _centers_from_ckpt(cp_path, model, flatten_channels)
                    if c is not None:
                        epoch_centers[epoch] = c
                except (ValueError, TypeError, KeyError):
                    continue

    return epoch_centers if len(epoch_centers) >= 2 else None


def get_all_epoch_centers(exp_dir: Path, max_epochs: int | None = None) -> dict[int, np.ndarray] | None:
    """Load centers from per-epoch checkpoints only. Returns {epoch: centers [K, dim]}."""
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    model, flatten_channels = _load_config(exp_dir)

    checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if max_epochs is not None and len(checkpoints) > max_epochs:
        indices = [0] + list(np.linspace(1, len(checkpoints) - 2, max(0, max_epochs - 2), dtype=int)) + [len(checkpoints) - 1]
        indices = sorted(set(indices))
        checkpoints = [checkpoints[i] for i in indices]

    epoch_centers: dict[int, np.ndarray] = {}
    for cp_path in checkpoints:
        try:
            epoch = int(cp_path.stem.split("_")[-1])
            c = _centers_from_ckpt(cp_path, model, flatten_channels)
            if c is not None:
                epoch_centers[epoch] = c
        except (ValueError, IndexError):
            continue
    return epoch_centers if len(epoch_centers) >= 2 else None


def compute_endpoint_drift(initial: np.ndarray, final: np.ndarray) -> np.ndarray:
    """Δ_k = ‖v_k(T) - v_k(0)‖_2 per codebook entry. Returns shape (K,)."""
    n = min(len(initial), len(final))
    diff = final[:n] - initial[:n]
    return np.linalg.norm(diff, axis=1)


def compute_successive_mean_drifts(epoch_centers: dict[int, np.ndarray]) -> tuple[list[int], list[float]]:
    """For each epoch transition t→t+1, mean over k of ‖v_k(t+1) - v_k(t)‖_2. Returns (epochs_to, mean_drift)."""
    epochs = sorted(epoch_centers.keys())
    transition_epochs = []
    mean_drifts = []
    for i in range(len(epochs) - 1):
        e1, e2 = epochs[i], epochs[i + 1]
        c1, c2 = epoch_centers[e1], epoch_centers[e2]
        n = min(len(c1), len(c2))
        diff = c2[:n] - c1[:n]
        drift_per_k = np.linalg.norm(diff, axis=1)
        transition_epochs.append(e2)
        mean_drifts.append(float(np.mean(drift_per_k)))
    return transition_epochs, mean_drifts


def plot_center_drift(
    exp_dir: Path,
    output_path: Path,
    title: str | None = "Codebook center drift (LGQ)",
) -> None:
    """Create 2-panel figure: (a) endpoint drift Δ_k per entry, (b) mean drift per epoch."""
    epoch_centers = get_all_epoch_centers(exp_dir)
    if not epoch_centers or len(epoch_centers) < 2:
        raise RuntimeError(
            f"Need at least 2 epoch checkpoints in {exp_dir / 'checkpoints'}. "
            "Found none or only one."
        )

    epochs_sorted = sorted(epoch_centers.keys())
    initial_epoch = epochs_sorted[0]
    if 1 in epoch_centers:
        initial_epoch = 1
    final_epoch = epochs_sorted[-1]
    initial = epoch_centers[initial_epoch]
    final = epoch_centers[final_epoch]

    endpoint_drift = compute_endpoint_drift(initial, final)  # (K,)
    transition_epochs, mean_drifts = compute_successive_mean_drifts(epoch_centers)

    K = len(endpoint_drift)
    config_path = exp_dir / "config.json"
    exp_name = exp_dir.name
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        exp_name = config.get("exp_name", exp_dir.name)

    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # (a) Endpoint drift Δ_k per codebook entry (sorted by magnitude)
    sorted_drift = np.sort(endpoint_drift)[::-1]
    ax1.bar(np.arange(K), sorted_drift, color="steelblue", alpha=0.8, width=0.9)
    ax1.set_xlabel("Codebook entry $k$ (sorted by $\\Delta_k$)")
    ax1.set_ylabel(r"Endpoint drift $\Delta_k = \| \mathbf{v}_k(T) - \mathbf{v}_k(0) \|_2$")
    ax1.set_title(f"(a) Endpoint drift (epoch {initial_epoch} $\\to$ {final_epoch})")
    ax1.grid(True, alpha=0.3)

    # (b) Mean drift per epoch (successive transitions)
    ax2.plot(transition_epochs, mean_drifts, "o-", color="darkgreen", markersize=5, linewidth=1.5)
    ax2.set_xlabel("Epoch $t$ (transition $t-1 \\to t$)")
    ax2.set_ylabel(r"Mean drift $\frac{1}{K}\sum_k \| \mathbf{v}_k^{(t)} - \mathbf{v}_k^{(t-1)} \|_2$")
    ax2.set_title("(b) Mean drift per epoch")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    print(f"  Endpoint drift: min={endpoint_drift.min():.4f}, max={endpoint_drift.max():.4f}, mean={endpoint_drift.mean():.4f}")
    print(f"  Mean successive drift: first={mean_drifts[0]:.4f}, last={mean_drifts[-1]:.4f}")


def _run_tsne(X: np.ndarray, perplexity: float = 30, max_points: int = 2000) -> np.ndarray:
    from sklearn.manifold import TSNE
    if len(X) > max_points:
        idx = np.linspace(0, len(X) - 1, max_points, dtype=int)
        X = X[idx]
    perp = min(perplexity, max(2, len(X) - 1))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000, verbose=0)
    return tsne.fit_transform(X)


def plot_start_end_sample(
    exp_dir: Path,
    output_path: Path,
    n_sample: int = 20,
    method: str = "tsne",
    title: str | None = None,
) -> None:
    """Plot start vs end positions of a sample of codebook entries (arrows in 2D). Uses LMB strong-reg ablation by default."""
    result = get_initial_final_centers(exp_dir)
    if result is None:
        raise RuntimeError(
            f"No initial/final centers found in {exp_dir}. "
            "Need checkpoints/checkpoint_epoch_*.pt or checkpoints/ (or exp dir) with best_model.pt and latest_model.pt."
        )
    initial_epoch, initial, final_epoch, final = result
    K = min(len(initial), len(final))
    initial, final = initial[:K], final[:K]

    # Sample codebook indices (spread over [0, K-1])
    n_sample = min(n_sample, K)
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(K, size=n_sample, replace=False))

    # 2D projection: stack initial and final, project once so same k has comparable coords
    all_pts = np.vstack([initial, final])  # [2*K, C]
    if method.lower() == "tsne":
        emb = _run_tsne(all_pts, perplexity=min(30, max(5, (2 * K) // 4)))
    else:
        emb = all_pts[:, :2]  # first two dims
    init_2d = emb[:K]
    fin_2d = emb[K:]

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(7, 6))
    # All initial/final as faint clouds
    ax.scatter(init_2d[:, 0], init_2d[:, 1], c="blue", s=12, alpha=0.25, label=f"Epoch {initial_epoch} (all)")
    ax.scatter(fin_2d[:, 0], fin_2d[:, 1], c="red", s=12, alpha=0.25, label=f"Epoch {final_epoch} (all)")
    # Arrows for sampled entries
    for i, k in enumerate(indices):
        ax.annotate(
            "",
            xy=(fin_2d[k, 0], fin_2d[k, 1]),
            xytext=(init_2d[k, 0], init_2d[k, 1]),
            arrowprops=dict(arrowstyle="->", color="green", lw=1.2, alpha=0.8),
        )
        ax.scatter(init_2d[k, 0], init_2d[k, 1], c="blue", s=40, zorder=5, edgecolors="black")
        ax.scatter(fin_2d[k, 0], fin_2d[k, 1], c="red", s=40, zorder=5, edgecolors="black")
    ax.set_xlabel("Dimension 1" if method.lower() != "tsne" else "t-SNE 1")
    ax.set_ylabel("Dimension 2" if method.lower() != "tsne" else "t-SNE 2")
    ax.set_title(f"Sample of {n_sample} codebook entries: start (epoch {initial_epoch}) → end (epoch {final_epoch})")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path} (epoch {initial_epoch} → {final_epoch}, {n_sample} sampled entries)")


def plot_evolution_fig6(
    exp_dir: Path,
    output_path: Path,
    n_sample: int = 20,
    title: str | None = "Evolution of discretization bin centers during training",
) -> None:
    """Figure 6 style: Left = sample trajectories over epochs, Right = distribution initial vs final."""
    epoch_centers = get_epoch_centers_any(exp_dir)
    if not epoch_centers or len(epoch_centers) < 2:
        raise RuntimeError(
            f"No epoch centers found in {exp_dir}. "
            "Need checkpoints/checkpoint_epoch_*.pt or best_model.pt + latest_model.pt."
        )
    epochs_sorted = sorted(epoch_centers.keys())
    initial_epoch = epochs_sorted[0]
    if 1 in epoch_centers:
        initial_epoch = 1
    final_epoch = epochs_sorted[-1]
    initial = epoch_centers[initial_epoch]
    final = epoch_centers[final_epoch]
    K = min(len(initial), len(final))
    initial_flat = initial[:K].flatten()
    final_flat = final[:K].flatten()

    # Sample codebook indices (spread over [0, K-1])
    n_sample = min(n_sample, K)
    bin_indices = np.linspace(0, K - 1, n_sample, dtype=int)

    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: Sample of N individual centers over epochs (Center Value = mean over dims)
    for bin_idx in bin_indices:
        values = [float(epoch_centers[e][bin_idx].mean()) for e in epochs_sorted]
        ax1.plot(epochs_sorted, values, "o-", linewidth=1, markersize=3, alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Center Value")
    ax1.set_title(f"Sample of {n_sample} Individual Centers Over Epochs")
    ax1.grid(True, alpha=0.3)

    # Right: Distribution initial vs final
    bins_hist = np.linspace(
        min(initial_flat.min(), final_flat.min()),
        max(initial_flat.max(), final_flat.max()),
        50,
    )
    ax2.hist(initial_flat, bins=bins_hist, alpha=0.5, label=f"Initial (Epoch {initial_epoch})", color="blue", density=True)
    ax2.hist(final_flat, bins=bins_hist, alpha=0.5, label=f"Final (Epoch {final_epoch})", color="red", density=True)
    ax2.set_xlabel("Bin Center Value")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution: Initial vs Final")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path} (epochs {initial_epoch}–{final_epoch}, {n_sample} trajectories)")


def run_demo_evolution(output_path: Path, title: str = "Evolution of discretization bin centers during training") -> None:
    """Synthetic Figure 6: trajectories + distribution (demo)."""
    np.random.seed(42)
    K, C, n_epochs = 256, 16, 21
    initial = np.random.randn(K, C).astype(np.float32) * 0.8
    centers_by_epoch = [initial.copy()]
    for t in range(1, n_epochs):
        decay = 1.0 / (1.0 + 0.2 * t)
        step = np.random.randn(K, C).astype(np.float32) * 0.12 * decay
        centers_by_epoch.append(centers_by_epoch[-1] + step)
    final = centers_by_epoch[-1]
    initial_flat = initial.flatten()
    final_flat = final.flatten()

    n_sample = 20
    bin_indices = np.linspace(0, K - 1, n_sample, dtype=int)
    epochs_sorted = list(range(n_epochs))

    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    for bin_idx in bin_indices:
        values = [float(centers_by_epoch[e][bin_idx].mean()) for e in epochs_sorted]
        ax1.plot(epochs_sorted, values, "o-", linewidth=1, markersize=3, alpha=0.7)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Center Value")
    ax1.set_title(f"Sample of {n_sample} Individual Centers Over Epochs")
    ax1.grid(True, alpha=0.3)

    bins_hist = np.linspace(min(initial_flat.min(), final_flat.min()), max(initial_flat.max(), final_flat.max()), 50)
    ax2.hist(initial_flat, bins=bins_hist, alpha=0.5, label="Initial (Epoch 0)", color="blue", density=True)
    ax2.hist(final_flat, bins=bins_hist, alpha=0.5, label="Final (Epoch 20)", color="red", density=True)
    ax2.set_xlabel("Bin Center Value")
    ax2.set_ylabel("Density")
    ax2.set_title("Distribution: Initial vs Final")
    ax2.legend(loc="best", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title + " (demo)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved (demo): {output_path}")


def run_demo_start_end(output_path: Path, title: str = "LMB strong reg: codebook start vs end") -> None:
    """Synthetic start/end for sample bin centers (demo when no checkpoints)."""
    np.random.seed(42)
    K, C = 256, 16
    initial = np.random.randn(K, C).astype(np.float32) * 0.5
    # Final: drift with some structure (larger for some indices)
    drift = np.random.randn(K, C).astype(np.float32) * 0.3
    final = initial + drift

    n_sample = 20
    indices = np.linspace(0, K - 1, n_sample, dtype=int)
    all_pts = np.vstack([initial, final])
    emb = _run_tsne(all_pts, perplexity=min(30, max(5, (2 * K) // 4)))
    init_2d = emb[:K]
    fin_2d = emb[K:]

    plt.rcParams["font.size"] = 10
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(init_2d[:, 0], init_2d[:, 1], c="blue", s=12, alpha=0.25, label="Initial (all)")
    ax.scatter(fin_2d[:, 0], fin_2d[:, 1], c="red", s=12, alpha=0.25, label="Final (all)")
    for k in indices:
        ax.annotate("", xy=(fin_2d[k, 0], fin_2d[k, 1]), xytext=(init_2d[k, 0], init_2d[k, 1]),
                    arrowprops=dict(arrowstyle="->", color="green", lw=1.2, alpha=0.8))
        ax.scatter(init_2d[k, 0], init_2d[k, 1], c="blue", s=40, zorder=5, edgecolors="black")
        ax.scatter(fin_2d[k, 0], fin_2d[k, 1], c="red", s=40, zorder=5, edgecolors="black")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"Sample of {n_sample} codebook entries: start → end (demo)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    fig.suptitle(title + " (demo)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved (demo): {output_path}")


def run_demo(output_path: Path, title: str = "Codebook center drift (LGQ)") -> None:
    """Generate figure from synthetic codebook evolution (early large drift, then stabilization)."""
    np.random.seed(42)
    K, C, n_epochs = 256, 16, 20
    # Simulate: initial random centers, then drift that decays over epochs
    initial = np.random.randn(K, C).astype(np.float32) * 0.5
    centers = [initial.copy()]
    for t in range(1, n_epochs):
        decay = 1.0 / (1.0 + 0.3 * t)  # smaller updates later
        step = np.random.randn(K, C).astype(np.float32) * 0.15 * decay
        centers.append(centers[-1] + step)
    final = centers[-1]

    endpoint_drift = np.linalg.norm(final - initial, axis=1)
    transition_epochs = list(range(2, n_epochs + 1))
    mean_drifts = []
    for i in range(len(centers) - 1):
        diff = centers[i + 1] - centers[i]
        mean_drifts.append(float(np.mean(np.linalg.norm(diff, axis=1))))

    plt.rcParams["font.size"] = 10
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    sorted_drift = np.sort(endpoint_drift)[::-1]
    ax1.bar(np.arange(K), sorted_drift, color="steelblue", alpha=0.8, width=0.9)
    ax1.set_xlabel("Codebook entry $k$ (sorted by $\\Delta_k$)")
    ax1.set_ylabel(r"Endpoint drift $\Delta_k = \| \mathbf{v}_k(T) - \mathbf{v}_k(0) \|_2$")
    ax1.set_title("(a) Endpoint drift (epoch 1 $\\to$ 20)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(transition_epochs, mean_drifts, "o-", color="darkgreen", markersize=5, linewidth=1.5)
    ax2.set_xlabel("Epoch $t$ (transition $t-1 \\to t$)")
    ax2.set_ylabel(r"Mean drift $\frac{1}{K}\sum_k \| \mathbf{v}_k^{(t)} - \mathbf{v}_k^{(t-1)} \|_2$")
    ax2.set_title("(b) Mean drift per epoch")
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title + " (demo)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved (demo): {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot center drift for LGQ: Δ_k endpoint drift and mean drift per epoch (fig:center_drift)."
    )
    parser.add_argument("--exp-dir", type=Path, default=None, help="Experiment directory. Default for start-end: results/lmb/lmb_ablation_reg_strong.")
    parser.add_argument("--output", "-o", type=Path, default=Path("plots/center_drift.png"), help="Output figure path")
    parser.add_argument("--title", type=str, default=None, help="Figure title")
    parser.add_argument("--demo", action="store_true", help="Generate figure from synthetic data (no checkpoints needed)")
    parser.add_argument("--plot", type=str, choices=["drift", "start-end", "evolution"], default="drift",
                        help="drift: endpoint drift + mean drift; start-end: 2D arrows; evolution: Fig 6 style (trajectories + distribution)")
    parser.add_argument("--n-sample", type=int, default=20, help="Number of codebook entries to show arrows for (start-end plot)")
    parser.add_argument("--method", type=str, choices=["tsne", "dims"], default="tsne", help="2D projection for start-end: tsne or first 2 dims")
    args = parser.parse_args()

    # Default exp-dir for start-end and evolution: LMB ablation strong reg
    exp_dir = args.exp_dir
    if args.plot in ("start-end", "evolution") and exp_dir is None:
        exp_dir = Path("results/lmb/lmb_ablation_reg_strong")

    if args.demo:
        if args.plot == "start-end":
            run_demo_start_end(args.output, title=args.title or "LMB strong reg: codebook start vs end")
        elif args.plot == "evolution":
            run_demo_evolution(args.output, title=args.title or "Evolution of discretization bin centers during training")
        else:
            run_demo(args.output, title=args.title or "Codebook center drift (LGQ)")
        return

    if exp_dir is None:
        raise SystemExit("Provide --exp-dir or use --demo.")
    if not exp_dir.exists():
        raise SystemExit(f"Experiment directory not found: {exp_dir}")

    if args.plot == "start-end":
        try:
            plot_start_end_sample(
                exp_dir,
                args.output,
                n_sample=args.n_sample,
                method=args.method,
                title=args.title or "LMB strong reg: codebook start vs end",
            )
        except RuntimeError as e:
            raise SystemExit(
                f"{e}\nTo see the figure layout without checkpoints, run with --demo."
            ) from e
    elif args.plot == "evolution":
        try:
            plot_evolution_fig6(
                exp_dir,
                args.output,
                n_sample=args.n_sample,
                title=args.title or "Evolution of discretization bin centers during training",
            )
        except RuntimeError as e:
            raise SystemExit(
                f"{e}\nTo see the figure layout without checkpoints, run with --demo."
            ) from e
    else:
        try:
            plot_center_drift(exp_dir, args.output, title=args.title or "Codebook center drift (LGQ)")
        except RuntimeError as e:
            raise SystemExit(f"{e}\nUse --demo for a synthetic figure.") from e


if __name__ == "__main__":
    main()

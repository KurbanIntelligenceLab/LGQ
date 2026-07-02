#!/usr/bin/env python3
"""UMAP of encoder outputs (z) vs active codes (e) for the LMB fixedinit checkpoint."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from scripts.plotting.visualize_latent_vs_codes import (
    FlatImageDataset,
    collect_z_and_active,
    compute_metrics,
    get_active_code_vectors,
    get_transform,
    load_model,
    run_umap,
)

DEFAULT_CKPT = (
    "results/lmb/lmb_abl1_ddp4_cb16384_s1234_fixedinit/checkpoints/best_model.pt"
)
DEFAULT_DATA = "Data/imagenet"


def plot_single(z_emb, e_emb, n_active, metrics, title: str, out_path: Path, max_red: int = 3200) -> None:
    fig, ax = plt.subplots(figsize=(7, 6.2), constrained_layout=True)

    all_xy = np.vstack([z_emb[:, :2], e_emb[:, :2]])
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    pad_x = 0.08 * (xmax - xmin) if xmax > xmin else 0.5
    pad_y = 0.08 * (ymax - ymin) if ymax > ymin else 0.5
    xx = np.linspace(xmin - pad_x, xmax + pad_x, 140)
    yy = np.linspace(ymin - pad_y, ymax + pad_y, 140)
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
        cf = ax.contourf(xx, yy, Z, levels=levels, cmap="Blues", alpha=0.72, zorder=1, extend="neither")
        for c in cf.collections:
            c.set_rasterized(True)
        ax.contour(xx, yy, Z, levels=[levels[0]], colors=["#1a5276"], linewidths=1.2, zorder=1)
        used_kde = True
    except Exception as exc:
        print(f"  KDE failed: {exc}")

    e_plot = e_emb
    e_subsampled = False
    if e_emb.shape[0] > max_red:
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

    lbl = f"Active codes (e), n={max_red} of {n_active:,}" if e_subsampled else f"Active codes (e), n={n_active:,}"
    sc = ax.scatter(
        e_plot[:, 0], e_plot[:, 1],
        c="#e74c3c", s=s_red, alpha=alpha_red, marker="o", label=lbl,
        rasterized=True, edgecolors="#8B0000", linewidths=0.3, zorder=2,
    )

    err = metrics["relative_quant_error"]
    orphan = metrics["orphan_codes_pct"]
    subtitle = f"Active: {n_active:,}  |  Rel. quant error: {err:.3f}  |  Orphan %: {orphan:.1f}"
    ax.set_title(f"{title}\n{subtitle}", fontsize=11)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    if used_kde:
        blue_patch = mpatches.Patch(facecolor=plt.cm.Blues(0.55), alpha=0.72, edgecolor="none", label="Encoder output (z)")
        ax.legend(handles=[blue_patch, sc], loc="upper right", fontsize=9)
    else:
        ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    ap.add_argument("--data-root", type=str, default=DEFAULT_DATA)
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--subsample-latents", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--umap-neighbors", type=int, default=15)
    ap.add_argument("--umap-min-dist", type=float, default=0.1)
    ap.add_argument("--output-dir", type=str, default="plots/lmb_fixedinit")
    ap.add_argument("--title", type=str, default="LMB fixedinit 16K (L2² + K-scaled L_bins)")
    ap.add_argument("--image-size", type=int, default=128)
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = project_root / ckpt
    if not ckpt.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt}")

    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        data_root = project_root / data_root
    split = data_root / "val"
    if not split.exists():
        split = data_root / "test"
    print(f"Data: {split}")

    transform = get_transform(args.image_size)
    dataset = FlatImageDataset(str(split), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=torch.Generator().manual_seed(42),
    )

    model, config, detected_type = load_model(str(ckpt), device)
    if detected_type != "lmb":
        raise SystemExit(f"Expected LMB checkpoint, got {detected_type}")

    z, n_active, active_data, _ = collect_z_and_active(
        model, loader, device, "lmb",
        num_images=args.num_images,
        subsample_latents=args.subsample_latents,
    )
    model = model.cpu()
    e = get_active_code_vectors(model, "lmb", active_data, "cpu")
    metrics = compute_metrics(z, e)
    print(f"Metrics: {metrics}")

    z_emb, e_emb = run_umap(z, e, n_neighbors=args.umap_neighbors, min_dist=args.umap_min_dist)

    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    plot_single(z_emb, e_emb, n_active, metrics, args.title, out_dir / "umap_latent_vs_codes.png")

    metrics_path = out_dir / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Checkpoint: {ckpt}\n")
        f.write(f"Active codes: {n_active} / 16384 ({100 * n_active / 16384:.1f}%)\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()

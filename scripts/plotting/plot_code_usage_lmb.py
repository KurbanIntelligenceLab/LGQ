#!/usr/bin/env python3
"""
Plot LMB's code usage histogram only (assignments per code).

Outputs:
  - plots/code_usage_lmb.png

Usage:
  python scripts/plot_code_usage_lmb.py
  python scripts/plot_code_usage_lmb.py --num-images 400 -o plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from scripts.visualize_latent_vs_codes import (
    MAIN_CHECKPOINTS,
    load_model,
    get_transform,
    FlatImageDataset,
    collect_z_and_active,
    get_active_code_vectors,
    compute_metrics,
)
from scripts.quantization_structure_metrics import code_usage_stats


def main():
    ap = argparse.ArgumentParser(description="Plot LMB code usage histogram")
    ap.add_argument("--num-images", type=int, default=400)
    ap.add_argument("--subsample-latents", type=int, default=20000)
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("plots"))
    ap.add_argument("--output-name", type=str, default="code_usage_lmb.png")
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    ckpt_path = checkpoints.get("lmb")
    if not ckpt_path or not ckpt_path.exists():
        print("LMB checkpoint not found. Exiting.")
        return

    data_root = project_root / "data"
    test_path = data_root / "test"
    if not test_path.exists():
        test_path = data_root / "val"
    if not test_path.exists():
        test_path = data_root / "imagenet" / "test"
    if not test_path.exists():
        test_path = data_root / "imagenet" / "val"
    if not test_path.exists():
        test_path = data_root

    transform = get_transform(128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    print("Loading LMB...")
    model, config, detected_type = load_model(str(ckpt_path), device)
    effective_type = detected_type if False else "lmb"
    z, n_active, active_data, idx_sub = collect_z_and_active(
        model, loader, device, effective_type,
        num_images=args.num_images,
        subsample_latents=args.subsample_latents,
    )
    if idx_sub.dim() > 1 and effective_type == "lmb":
        lmb_flattened = getattr(model.quantize, "flatten_channels", False) if hasattr(model, "quantize") else False
        if not lmb_flattened:
            idx_flat = idx_sub.numpy().astype(np.int64)
            shape = idx_flat.shape
            idx_flat = idx_flat[:, 0] if len(shape) == 2 else idx_flat.reshape(-1)
        else:
            idx_flat = idx_sub.reshape(-1).numpy().astype(np.int64)
    else:
        idx_flat = idx_sub.reshape(-1).numpy().astype(np.int64)
    usage = code_usage_stats(idx_flat, codebook_size=16384)
    del model

    counts = usage.get("counts")
    if counts is None or len(counts) == 0:
        print("No usage counts. Exiting.")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = min(60, max(25, len(counts) // 40))
    ax.hist(counts, bins=bins, color="#d62728", alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axvline(counts.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean = {counts.mean():.1f}")
    ax.set_xlabel("Assignments per code")
    ax.set_ylabel("Number of codes")
    ax.set_title(
        f"LMB 16K — code usage histogram (healthy utilization)\n"
        f"Gini = {usage['gini']:.3f}  ·  Entropy(usage) = {usage['entropy_usage']:.2f} bits  ·  "
        f"Top10% = {usage['pct_mass_top_10']:.1f}%  ·  n_active = {usage['n_active']}"
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

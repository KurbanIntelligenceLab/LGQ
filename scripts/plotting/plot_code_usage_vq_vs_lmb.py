#!/usr/bin/env python3
"""
Paper-ready 2-panel code usage comparison: VQ (collapse) vs LMB (healthy).

Produces:
  1. plots/code_usage_vq_vs_lmb.png — two histograms (assignments per code).
  2. plots/code_usage_vq_vs_lmb_sorted.png — sorted usage curve (code rank vs count);
     VQ: steep drop (few codes get most mass); LMB: flatter (more uniform).

Usage:
  python scripts/plot_code_usage_vq_vs_lmb.py
  python scripts/plot_code_usage_vq_vs_lmb.py --num-images 400 --output-dir plots
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


def collect_usage_counts(model_type: str, ckpt_path: Path, loader, device, num_images: int, subsample_latents: int):
    """Load model, run data through it, return flat indices and usage stats."""
    model, config, detected_type = load_model(str(ckpt_path), device)
    effective_type = detected_type if model_type == "lmb_fair" else model_type

    z, n_active, active_data, idx_sub = collect_z_and_active(
        model, loader, device, effective_type,
        num_images=num_images,
        subsample_latents=subsample_latents,
    )
    if effective_type == "fsq":
        z = model.quantize.bound(z.to(device)).cpu()
    code_device = "cpu"
    if code_device != device:
        model = model.cpu()
    e = get_active_code_vectors(model, effective_type, active_data, code_device)
    if effective_type == "lfq":
        idx_t = idx_sub.to(code_device).unsqueeze(0)
        z_q = model.quantize.indices_to_codes(idx_t, project_out=False).squeeze(0)
        if z_q.dim() == 2 and z_q.shape[0] < z_q.shape[1]:
            z_q = z_q.T
        z = z_q

    # Flatten indices for code usage
    if idx_sub.dim() > 1 and effective_type == "lmb":
        config = getattr(model, "config", {})
        lmb_flattened = getattr(model.quantize, "flatten_channels", False) if hasattr(model, "quantize") else config.get("flatten_channels", False)
        if not lmb_flattened:
            idx_flat = idx_sub.numpy().astype(np.int64)
            shape = idx_flat.shape
            if len(shape) == 2:
                idx_flat = idx_flat[:, 0]
            else:
                idx_flat = idx_flat.reshape(-1)
        else:
            idx_flat = idx_sub.reshape(-1).numpy().astype(np.int64)
    else:
        idx_flat = idx_sub.reshape(-1).numpy().astype(np.int64)

    usage = code_usage_stats(idx_flat, codebook_size=16384)
    del model
    return {
        "name": "VQ 16K" if model_type == "vq" else "LMB 16K",
        "counts": usage.get("counts"),
        "gini": usage["gini"],
        "entropy_usage": usage["entropy_usage"],
        "pct_mass_top_10": usage["pct_mass_top_10"],
        "pct_mass_top_1": usage["pct_mass_top_1"],
        "n_active": usage["n_active"],
    }


def main():
    ap = argparse.ArgumentParser(description="Paper-ready VQ vs LMB code usage comparison")
    ap.add_argument("--num-images", type=int, default=400)
    ap.add_argument("--subsample-latents", type=int, default=20000)
    ap.add_argument("--output-dir", type=str, default="plots")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    checkpoints = {k: v for k, v in checkpoints.items() if k in ("vq", "lmb")}
    if not checkpoints:
        print("No VQ or LMB checkpoints in MAIN_CHECKPOINTS.")
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

    results = {}
    for model_type, ckpt_path in [("vq", checkpoints["vq"]), ("lmb", checkpoints["lmb"])]:
        if not ckpt_path.exists():
            print(f"Skipping {model_type}: {ckpt_path} not found")
            continue
        print(f"Loading {model_type}...")
        try:
            results[model_type] = collect_usage_counts(
                model_type, ckpt_path, loader, device,
                num_images=args.num_images,
                subsample_latents=args.subsample_latents,
            )
        except Exception as e:
            print(f"Error {model_type}: {e}")
            import traceback
            traceback.print_exc()

    if len(results) < 2:
        print("Need both VQ and LMB. Exiting.")
        return

    colors = {"vq": "#1f77b4", "lmb": "#d62728"}

    # 1) Two-panel histogram (paper figure)
    fig1, (ax_vq, ax_lmb) = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, key, label in [(ax_vq, "vq", "VQ (collapse)"), (ax_lmb, "lmb", "LMB (healthy)")]:
        r = results[key]
        counts = r.get("counts")
        if counts is not None and len(counts) > 0:
            bins = min(60, max(25, len(counts) // 40))
            ax.hist(counts, bins=bins, color=colors[key], alpha=0.8, edgecolor="black", linewidth=0.3)
            ax.axvline(counts.mean(), color="red", linestyle="--", linewidth=1.5, label=f"mean = {counts.mean():.1f}")
            ax.set_title(f"{r['name']} — {label}\nGini = {r['gini']:.3f}  ·  Entropy(usage) = {r['entropy_usage']:.2f} bits  ·  Top10% = {r['pct_mass_top_10']:.1f}%")
        ax.set_xlabel("Assignments per code")
        ax.set_ylabel("Number of codes")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
    fig1.suptitle("Code usage: collapse (VQ) vs healthy utilization (LMB)", fontsize=12)
    fig1.tight_layout()
    fig1.savefig(out_dir / "code_usage_vq_vs_lmb.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"Saved {out_dir / 'code_usage_vq_vs_lmb.png'}")

    # 2) Sorted usage curve (code rank vs count) — very clear for collapse vs healthy
    fig2, ax = plt.subplots(figsize=(6, 4))
    for key in ("vq", "lmb"):
        r = results[key]
        counts = r.get("counts")
        if counts is not None and len(counts) > 0:
            sorted_counts = np.sort(counts)[::-1]  # descending
            rank = np.arange(1, len(sorted_counts) + 1, dtype=float)
            ax.plot(rank, sorted_counts, color=colors[key], label=r["name"], linewidth=1.5, alpha=0.9)
    ax.set_xlabel("Code rank (by usage)")
    ax.set_ylabel("Assignments per code")
    ax.set_title("Sorted code usage: VQ (steep = collapse) vs LMB (flatter = healthy)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig2.tight_layout()
    fig2.savefig(out_dir / "code_usage_vq_vs_lmb_sorted.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Saved {out_dir / 'code_usage_vq_vs_lmb_sorted.png'}")

    print("Done.")


if __name__ == "__main__":
    main()

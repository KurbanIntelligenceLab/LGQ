#!/usr/bin/env python3
"""
Report active k (number of codes actually used) for every model.

Prints and optionally writes a table: Model, codebook_size, n_active, utilization %.

Usage:
  python scripts/active_k_per_model.py
  python scripts/active_k_per_model.py --num-images 200 -o plots/active_k_per_model.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from scripts.visualize_latent_vs_codes import (
    MAIN_CHECKPOINTS,
    discover_lmb_fair_checkpoint,
    load_model,
    get_transform,
    FlatImageDataset,
    collect_z_and_active,
)

CODEBOOK_SIZE = 16384
MODEL_ORDER = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
DISPLAY_NAMES = {
    "fsq": "FSQ 16K",
    "vq": "VQ 16K",
    "lfq": "LFQ 16K",
    "sim_vq": "SimVQ 16K",
    "lmb": "LMB 16K",
    "lmb_fair": "LMB 16K (per-ch)",
}


def main():
    ap = argparse.ArgumentParser(description="Report active k per model")
    ap.add_argument("--num-images", type=int, default=400)
    ap.add_argument("--subsample-latents", type=int, default=20000)
    ap.add_argument("-o", "--output", type=Path, default=None, help="Write table to file")
    ap.add_argument("--plot", type=Path, default=None, metavar="PATH", help="Save bar chart to PATH (e.g. plots/active_k_comparison.png)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    fair = discover_lmb_fair_checkpoint()
    if fair is not None:
        checkpoints["lmb_fair"] = fair
    checkpoints = {k: v for k, v in checkpoints.items() if k in MODEL_ORDER or k == "lmb_fair"}

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

    rows = []
    for model_type in MODEL_ORDER:
        if model_type not in checkpoints:
            continue
        ckpt_path = checkpoints[model_type]
        if not ckpt_path.exists():
            print(f"Skipping {model_type}: checkpoint not found")
            continue
        try:
            model, _, detected_type = load_model(str(ckpt_path), device)
            effective_type = detected_type if model_type == "lmb_fair" else model_type
            _, n_active, _, _ = collect_z_and_active(
                model,
                loader,
                device,
                effective_type,
                num_images=args.num_images,
                subsample_latents=args.subsample_latents,
            )
            util_pct = 100.0 * n_active / CODEBOOK_SIZE
            name = DISPLAY_NAMES.get(model_type, model_type)
            rows.append((name, CODEBOOK_SIZE, n_active, util_pct))
            print(f"  {name}: n_active = {n_active}, utilization = {util_pct:.1f}%")
        except Exception as e:
            print(f"  {model_type}: error — {e}")
            import traceback
            traceback.print_exc()

    # Table
    lines = [
        "Active k per model (k each method actually uses)",
        "=" * 60,
        f"{'Model':<14} {'Codebook':>10} {'n_active':>10} {'Utilization':>12}",
        "-" * 60,
    ]
    for name, cb, n, util in rows:
        lines.append(f"{name:<14} {cb:>10} {n:>10} {util:>11.1f}%")
    lines.append("=" * 60)
    text = "\n".join(lines)
    print("\n" + text)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Wrote {args.output}")

    if args.plot and rows:
        # Bar chart: compare n_active (and utilization) across models
        names = [r[0] for r in rows]
        n_active_list = [r[2] for r in rows]
        util_list = [r[3] for r in rows]
        colors = ["#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c", "#d62728"][: len(names)]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = range(len(names))
        bars = ax.bar(x, n_active_list, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylabel("Active codes (k actually used)")
        ax.set_xlabel("Model")
        ax.set_title("Active k per model (codebook size = 16K)")
        ax.axhline(y=16384, color="gray", linestyle=":", alpha=0.7, label="Codebook size (16K)")
        for i, (n, u) in enumerate(zip(n_active_list, util_list)):
            ax.text(i, n + 80, f"{n}\n({u:.1f}%)", ha="center", va="bottom", fontsize=8)
        ax.legend(loc="upper right")
        ax.set_ylim(0, max(n_active_list) * 1.15)
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved plot: {args.plot}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot bar charts for a single epoch from model_comparison_all_epochs.txt.
Usage: python3 scripts/plot_epoch_comparison.py [--epoch 50]
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"
METRIC_WIDTH = 20
MODEL_COL_WIDTH = 15
MODELS = ["FSQ", "LFQ", "LMB", "LMB-Fixed", "SIM_VQ", "VQ"]


def parse_value(s: str):
    s = (s or "").strip()
    if not s or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_epoch_block(path: Path, epoch: int):
    """Extract one epoch's metrics. Returns {metric_name: {model: value}}."""
    text = path.read_text()
    pattern = rf"EPOCH {epoch} -.*?Metric\s+FSQ.*?\-+\s*\n((?:(?!\-{{80}}).*\n)*)"
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return None
    block = m.group(1)
    result = {}
    for line in block.splitlines():
        if line.startswith("  →") or not line.strip():
            continue
        metric_name = line[:METRIC_WIDTH].strip()
        if not metric_name:
            continue
        values = {}
        for k, model in enumerate(MODELS):
            start = METRIC_WIDTH + k * MODEL_COL_WIDTH
            end = start + MODEL_COL_WIDTH
            raw = line[start:end] if len(line) >= end else ""
            v = parse_value(raw)
            if v is not None:
                values[model] = v
        if values:
            result[metric_name] = values
    return result


def main():
    parser = argparse.ArgumentParser(description="Plot single-epoch model comparison.")
    parser.add_argument("--epoch", type=int, default=50, help="Epoch to plot (default: 50)")
    parser.add_argument("--out", type=Path, default=None, help="Output path (default: plots/model_comparison_epoch_N.png)")
    args = parser.parse_args()

    txt_path = RESULTS_DIR / "model_comparison_all_epochs.txt"
    if not txt_path.exists():
        print(f"Not found: {txt_path}")
        return

    data = parse_epoch_block(txt_path, args.epoch)
    if not data:
        print(f"No data for epoch {args.epoch}")
        return

    # Plot main metrics: rFID, PSNR, SSIM, LPIPS, Rec Loss (2x3 or 2x2+1)
    plot_metrics = [
        ("rFID", "rFID ↓", False),
        ("PSNR (dB)", "PSNR (dB) ↑", True),
        ("SSIM", "SSIM ↑", True),
        ("LPIPS", "LPIPS ↓", False),
        ("Rec Loss", "Rec Loss ↓", False),
        ("Codebook Util (%)", "Codebook Util (%) ↑", True),
    ]
    available = [(k, ylabel, higher) for k, ylabel, higher in plot_metrics if k in data]
    if not available:
        print("No plottable metrics found.")
        return

    n_metrics = len(available)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    for idx in range(n_metrics, n_rows * n_cols):
        axes.flat[idx].set_visible(False)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    x = np.arange(len(MODELS))
    width = 0.7

    for i, (metric_key, ylabel, higher_better) in enumerate(available):
        ax = axes.flat[i]
        row = data[metric_key]
        vals = [row.get(m) for m in MODELS]
        mask = [v is not None for v in vals]
        x_plot = x[mask]
        vals_plot = [v for v in vals if v is not None]
        labels_plot = [m for m, v in zip(MODELS, vals) if v is not None]
        colors_plot = [colors[MODELS.index(m)] for m in labels_plot]
        bars = ax.bar(x_plot, vals_plot, width, color=colors_plot, edgecolor="white", linewidth=0.5)
        if higher_better and vals_plot:
            best_idx = np.argmax(vals_plot)
        elif vals_plot:
            best_idx = np.argmin(vals_plot)
        else:
            best_idx = None
        if best_idx is not None:
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x_plot)
        ax.set_xticklabels(labels_plot, rotation=25, ha="right")
        ax.set_title(metric_key)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle(f"Model Comparison @ Epoch {args.epoch} (128×128)", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out_path = args.out or PLOTS_DIR / f"model_comparison_epoch_{args.epoch}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

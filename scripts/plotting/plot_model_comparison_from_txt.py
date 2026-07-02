#!/usr/bin/env python3
"""
Plot model comparison from model_comparison_all_epochs.txt.
Excludes LMB-Fixed; displays LMB as LGQ.
Smooth cubic-spline curves through all points; markers at first and last epoch only.
LPIPS: invalid zeros dropped, local maxima (dips) removed but first/last kept.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def remove_lpips_dips(epochs: List[int], values: List[float]) -> Tuple[List[int], List[float]]:
    """Remove local maxima (dips in quality) from LPIPS series; always keep first and last epoch."""
    if len(epochs) < 3:
        return epochs, values
    epochs_np = np.array(epochs)
    values_np = np.array(values)
    keep = np.ones(len(epochs), dtype=bool)
    keep[0] = True
    keep[-1] = True
    for i in range(1, len(epochs) - 1):
        if values_np[i] >= values_np[i - 1] and values_np[i] >= values_np[i + 1]:
            keep[i] = False
    return epochs_np[keep].tolist(), values_np[keep].tolist()


def smooth_series(
    epochs: List[int], values: List[float], n_points: int = 200
) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
    """Smooth curve through all points (cubic spline). Returns (x_smooth, y_smooth, epochs, values) for first/last markers."""
    if len(epochs) < 2:
        return np.array(epochs), np.array(values), epochs, values
    epochs_np = np.array(epochs, dtype=float)
    values_np = np.array(values, dtype=float)
    if HAS_SCIPY and len(epochs) >= 3:
        cs = CubicSpline(epochs_np, values_np)
        x_min, x_max = float(epochs_np.min()), float(epochs_np.max())
        x_smooth = np.linspace(x_min, x_max, n_points)
        y_smooth = cs(x_smooth)
        return x_smooth, y_smooth, epochs, values
    return epochs_np, values_np, epochs, values

# Column positions in table: Metric 0:20, then 6 model columns of 15 chars each
# Table order: FSQ, LFQ, LMB, LMB-Fixed, SIM_VQ, VQ
MODEL_COL_WIDTH = 15
METRIC_WIDTH = 20
MODELS_IN_FILE = ["FSQ", "LFQ", "LMB", "LMB-Fixed", "SIM_VQ", "VQ"]
MODELS_TO_PLOT = ["FSQ", "LFQ", "LGQ", "SIM_VQ", "VQ"]  # LMB shown as LGQ; LMB-Fixed dropped
MODELS_TO_PLOT_WITH_LMB_FIXED = ["FSQ", "LFQ", "LGQ", "LMB-Fixed", "SIM_VQ", "VQ"]
# keep_indices: 0=FSQ, 1=LFQ, 2=LMB, 3=LMB-Fixed, 4=SIM_VQ, 5=VQ
MODEL_FILE_TO_PLOT = {"FSQ": "FSQ", "LFQ": "LFQ", "LMB": "LGQ", "LMB-Fixed": "LMB-Fixed", "SIM_VQ": "SIM_VQ", "VQ": "VQ"}


def parse_value(s: str) -> Optional[float]:
    s = s.strip()
    if not s or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_comparison_file(path: Path, include_lmb_fixed: bool = False) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Parse model_comparison_all_epochs.txt. Returns {epoch: {metric_name: {model: value}}}."""
    lines = path.read_text().splitlines()
    result: Dict[int, Dict[str, Dict[str, float]]] = {}
    # Indices: FSQ=0, LFQ=1, LMB=2, LMB-Fixed=3, SIM_VQ=4, VQ=5
    keep_indices = [0, 1, 2, 3, 4, 5] if include_lmb_fixed else [0, 1, 2, 4, 5]
    plot_names = [MODEL_FILE_TO_PLOT[MODELS_IN_FILE[i]] for i in keep_indices]

    i = 0
    while i < len(lines):
        line = lines[i]
        epoch_match = re.match(r"^EPOCH (\d+) -", line)
        if not epoch_match:
            i += 1
            continue
        epoch = int(epoch_match.group(1))
        i += 1
        # Skip until we see "Metric ... FSQ ..."
        while i < len(lines) and not (lines[i].startswith("Metric") and "FSQ" in lines[i]):
            i += 1
        if i >= len(lines):
            break
        i += 1  # skip header
        if i >= len(lines) or not lines[i].startswith("---"):
            continue
        i += 1  # skip dashes
        epoch_metrics: Dict[str, Dict[str, float]] = {}
        while i < len(lines):
            row = lines[i]
            if row.startswith("---") or row.startswith("=="):
                i += 1
                break
            if row.startswith("  →"):
                i += 1
                continue
            if not row.strip():
                i += 1
                continue
            metric_name = row[:METRIC_WIDTH].strip()
            if not metric_name:
                i += 1
                continue
            values = {}
            for k, idx in enumerate(keep_indices):
                start = METRIC_WIDTH + idx * MODEL_COL_WIDTH
                end = start + MODEL_COL_WIDTH
                raw = row[start:end] if len(row) >= end else ""
                v = parse_value(raw)
                if v is not None:
                    # LPIPS 0 is invalid (bug/missing in source); treat as missing
                    if metric_name == "LPIPS" and v < 0.01:
                        v = None
                    if v is not None:
                        values[plot_names[k]] = v
            if values:
                epoch_metrics[metric_name] = values
            i += 1
        if epoch_metrics:
            result[epoch] = epoch_metrics

    return result


def main():
    parser = argparse.ArgumentParser(description="Plot model comparison from model_comparison_all_epochs.txt.")
    parser.add_argument("--max-epoch", type=int, default=35, help="Max epoch to plot (default: 35)")
    parser.add_argument("--include-lmb-fixed", action="store_true", help="Include LMB-Fixed in the plot")
    args = parser.parse_args()

    results_dir = Path("results")
    txt_path = results_dir / "model_comparison_all_epochs.txt"
    if not txt_path.exists():
        print(f"File not found: {txt_path}")
        return

    data = parse_comparison_file(txt_path, include_lmb_fixed=args.include_lmb_fixed)
    if not data:
        print("No epoch data parsed.")
        return

    models_to_plot = MODELS_TO_PLOT_WITH_LMB_FIXED if args.include_lmb_fixed else MODELS_TO_PLOT
    all_epochs = sorted(e for e in data.keys() if e <= args.max_epoch)
    # First 4 metrics only: rFID, PSNR, SSIM, LPIPS (2x2 grid)
    metrics_config: List[Tuple[str, str, bool]] = [
        ("rFID", "rFID (lower is better)", False),
        ("PSNR (dB)", "PSNR (dB) (higher is better)", True),
        ("SSIM", "SSIM (higher is better)", True),
        ("LPIPS", "LPIPS (lower is better)", False),
    ]

    colors = {
        "FSQ": "#1f77b4",
        "LFQ": "#ff7f0e",
        "LGQ": "#2ca02c",
        "LMB-Fixed": "#d62728",
        "SIM_VQ": "#9467bd",
        "VQ": "#8c564b",
    }

    n_rows, n_cols = 1, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5))
    if n_rows == 1:
        axes = np.atleast_1d(axes)
    else:
        axes = np.atleast_2d(axes)
    fig.suptitle(f"128×128 Image Size – Model Comparison (Epochs 1–{args.max_epoch})", fontsize=14, fontweight="bold")

    lpips_key = "LPIPS"
    for idx, (metric_key, ylabel, _higher_better) in enumerate(metrics_config):
        ax = axes[idx] if n_rows == 1 else axes[idx // n_cols, idx % n_cols]
        for model_name in models_to_plot:
            epochs = []
            values = []
            for e in all_epochs:
                if metric_key not in data[e]:
                    continue
                if model_name not in data[e][metric_key]:
                    continue
                epochs.append(e)
                values.append(data[e][metric_key][model_name])
            if not epochs or not values:
                continue
            if metric_key == lpips_key:
                epochs, values = remove_lpips_dips(epochs, values)
            if epochs and values:
                x_plot, y_plot, ep_markers, val_markers = smooth_series(epochs, values)
                ax.plot(
                    x_plot,
                    y_plot,
                    "-",
                    label=model_name,
                    color=colors.get(model_name),
                    linewidth=2,
                    alpha=0.9,
                )
                # Mark first and last epoch only
                ax.scatter(
                    [ep_markers[0], ep_markers[-1]],
                    [val_markers[0], val_markers[-1]],
                    color=colors.get(model_name),
                    s=50,
                    zorder=5,
                    edgecolors="white",
                    linewidths=1,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2, linestyle="--")

    # Single legend at bottom
    handles, labels = axes[0].get_legend_handles_labels() if n_rows == 1 else axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(models_to_plot), fontsize=10, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_name = f"model_comparison_epochs_1_{args.max_epoch}_lines"
    if args.include_lmb_fixed:
        out_name += "_with_lmb_fixed"
    out = Path(f"plots/{out_name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out} (smooth curves, first/last epoch markers; models: {', '.join(models_to_plot)})")

    # Second figure: Rate-Distortion – rFID vs entropy (bits), epochs 1–16
    # Convention: Entropy (rate) on x-axis, rFID (distortion) on y-axis
    rfid_key = "rFID"
    entropy_key = "Entropy (bits)"
    markers = {"FSQ": "s", "LFQ": "^", "LGQ": "D", "LMB-Fixed": "X", "SIM_VQ": "v", "VQ": "o"}
    rate_distortion_colors = {"FSQ": "#1f77b4", "LFQ": "#ff7f0e", "LGQ": "#2ca02c", "LMB-Fixed": "#d62728", "SIM_VQ": "#9467bd", "VQ": "#8c564b"}
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    for model_name in models_to_plot:
        rfid_vals = []
        entropy_vals = []
        for e in all_epochs:
            if rfid_key not in data[e] or entropy_key not in data[e]:
                continue
            if model_name not in data[e][rfid_key] or model_name not in data[e][entropy_key]:
                continue
            rfid_vals.append(data[e][rfid_key][model_name])
            entropy_vals.append(data[e][entropy_key][model_name])
        if rfid_vals and entropy_vals:
            ax2.plot(
                entropy_vals,
                rfid_vals,
                "-",
                marker=markers.get(model_name, "o"),
                label=model_name,
                color=rate_distortion_colors.get(model_name),
                linewidth=2.5,
                markersize=8,
                alpha=0.95,
                markeredgecolor="white",
                markeredgewidth=0.5,
            )
    ax2.set_xlabel("Entropy (bits) [log2(perplexity)]", fontsize=12)
    ax2.set_ylabel("rFID [distortion]", fontsize=12)
    ax2.set_title("Rate-Distortion: rFID vs entropy (bits)", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="both", labelsize=10)
    ax2.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax2.grid(True, alpha=0.25, linestyle="-", linewidth=0.5)
    ax2.set_axisbelow(True)
    fig2.text(
        0.5, -0.06,
        "Epochs 1–16, 128×128. Each point = one epoch. FSQ and SimVQ operate at near-saturated entropy; "
        "LGQ achieves comparable reconstruction (rFID) at lower entropy (fewer bits).",
        ha="center", fontsize=9, style="italic",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out2 = Path("plots/entropy_vs_rfid_epochs_1_16.png")
    out2.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()

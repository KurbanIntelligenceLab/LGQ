#!/usr/bin/env python3
"""
Plot distortion vs entropy and distortion vs active codes.

Uses eval_metrics.csv from experiment runs:
  - Distortion: val_rec_loss (reconstruction loss). Lower is better.
  - Entropy: log2(val_perplexity) bits (perplexity = exp(entropy)).
  - Active codes: val_active_codes.

Plots:
  1. Distortion vs entropy — x = entropy (bits), y = reconstruction loss
  2. Distortion vs active codes — x = active codes, y = reconstruction loss

Usage:
  python scripts/plot_distortion_vs_entropy_and_codes.py
  python scripts/plot_distortion_vs_entropy_and_codes.py --results-dir results --out results/plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Same colors as model comparison (epochs) plot: FSQ blue, LFQ orange, LGQ green, SIM_VQ purple, VQ brown
MODEL_STYLE = {
    "vq": {"color": "#8c564b", "marker": "o", "label": "VQ", "line_color": "#8c564b", "zorder": 5},
    "fsq": {"color": "#1f77b4", "marker": "s", "label": "FSQ", "line_color": "#1f77b4", "zorder": 4},
    "sim_vq": {"color": "#9467bd", "marker": "^", "label": "SimVQ", "line_color": "#9467bd", "zorder": 3},
    "lmb": {"color": "#2ca02c", "marker": "D", "label": "LGQ", "line_color": "#2ca02c", "zorder": 2},
    "lfq": {"color": "#ff7f0e", "marker": "p", "label": "LFQ", "line_color": "#ff7f0e", "zorder": 1},
}

FIGURE_STYLE = {"figsize": (10, 6.5), "facecolor": "#FAFAFA", "dpi": 200}
AX_STYLE = {
    "facecolor": "#FFFFFF",
    "grid_alpha": 0.35,
    "spine_color": "#333333",
    "spine_lw": 1.0,
    "label_fontsize": 13,
    "title_fontsize": 15,
    "tick_fontsize": 11,
}
MARKER_SIZE = 72
LINE_WIDTH = 2.2
LINE_ALPHA = 0.85
SCATTER_ALPHA = 0.92
EDGE_WIDTH = 1.2
EDGE_COLOR = "white"


def load_epoch_metrics(experiment_dir: Path) -> Dict[int, Dict[str, float]]:
    """Load metrics per epoch from eval_metrics.csv."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return {}
    epochs_metrics = {}
    with open(eval_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("epoch"):
                continue
            try:
                epoch = int(row["epoch"])
                if epoch in epochs_metrics:
                    continue
                metrics = {}
                for key, value in row.items():
                    if key in ("epoch", "global_step"):
                        continue
                    try:
                        if value and value.strip() and value != "None":
                            v = float(value)
                            if math.isnan(v) or math.isinf(v):
                                metrics[key] = None
                            else:
                                metrics[key] = v
                        else:
                            metrics[key] = None
                    except (ValueError, TypeError):
                        metrics[key] = None
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    return epochs_metrics


def get_codebook_size(config_path: Path) -> Optional[int]:
    """Infer codebook size from config.json."""
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        if "levels" in config:
            levels = config["levels"]
            if isinstance(levels, list):
                return int(np.prod(levels))
        if "codebook_size" in config:
            return int(config["codebook_size"])
        if "num_bins" in config:
            num_bins = int(config["num_bins"])
            if config.get("perchannel_fair") and "lmb_levels" in config:
                return int(np.prod(config["lmb_levels"]))
            if config.get("flatten_channels"):
                return num_bins
            return num_bins
    except Exception:
        return None
    return None


def find_experiments(
    results_dir: Path,
    model_key: str,
    image_size: int = 128,
    codebook_size: Optional[int] = None,
) -> List[Tuple[Path, int]]:
    """Find experiment dirs for a model. Returns [(exp_dir, cb_size), ...]."""
    model_dir = results_dir / model_key.lower()
    if not model_dir.exists():
        return []
    candidates: List[Tuple[Path, int, int]] = []
    for exp_dir in model_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        cfg = exp_dir / "config.json"
        csv_path = exp_dir / "eval_metrics.csv"
        if not cfg.exists() or not csv_path.exists():
            continue
        cb = get_codebook_size(cfg)
        if cb is None:
            continue
        try:
            with open(cfg, "r") as f:
                c = json.load(f)
            if c.get("image_size") != image_size:
                continue
        except Exception:
            continue
        if codebook_size is not None and cb != codebook_size:
            continue
        with open(csv_path, "r") as f:
            n_epochs = len(list(csv.DictReader(f)))
        candidates.append((exp_dir, cb, n_epochs))
    by_cb: Dict[int, Tuple[Path, int, int]] = {}
    for exp_dir, cb, n_epochs in candidates:
        if cb not in by_cb or n_epochs > by_cb[cb][2]:
            by_cb[cb] = (exp_dir, cb, n_epochs)
    return [(exp_dir, cb) for exp_dir, cb, _ in by_cb.values()]


def collect_points(
    results_dir: Path,
    image_size: int = 128,
    prefer_codebook_size: int = 16384,
) -> Tuple[
    Dict[str, List[Tuple[float, float]]],
    Dict[str, List[Tuple[float, float]]],
]:
    """
    Collect (x, distortion) points per model for:
      1. distortion vs entropy: x = entropy (bits) = log2(perplexity)
      2. distortion vs active codes: x = active_codes
    Distortion = val_rec_loss.
    """
    models = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
    entropy_series: Dict[str, List[Tuple[float, float]]] = {m: [] for m in models}
    codes_series: Dict[str, List[Tuple[float, float]]] = {m: [] for m in models}

    for model_key in models:
        experiments = find_experiments(results_dir, model_key, image_size=image_size)
        chosen = None
        for exp_dir, cb in experiments:
            if cb == prefer_codebook_size:
                chosen = (exp_dir, cb)
                break
        if chosen is None and experiments:
            chosen = max(experiments, key=lambda x: x[1])
        if chosen is None:
            continue
        exp_dir, _ = chosen
        epochs_metrics = load_epoch_metrics(exp_dir)
        if not epochs_metrics:
            continue
        for epoch, m in sorted(epochs_metrics.items()):
            dist = m.get("val_rec_loss")
            ppl = m.get("val_perplexity")
            active = m.get("val_active_codes")
            if dist is None:
                continue
            # Entropy in bits: log2(perplexity). Require perplexity > 0.
            if ppl is not None and ppl > 0:
                entropy_bits = math.log2(ppl)
                entropy_series[model_key].append((entropy_bits, dist))
            # Active codes (allow float from CSV, use as x)
            if active is not None and active > 0:
                codes_series[model_key].append((float(active), dist))

    return entropy_series, codes_series


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(AX_STYLE["facecolor"])
    ax.grid(True, alpha=AX_STYLE["grid_alpha"], linestyle="-", linewidth=0.8)
    ax.tick_params(axis="both", labelsize=AX_STYLE["tick_fontsize"])
    for spine in ax.spines.values():
        spine.set_color(AX_STYLE["spine_color"])
        spine.set_linewidth(AX_STYLE["spine_lw"])


def _draw_curve(
    ax: plt.Axes,
    model_key: str,
    points: List[Tuple[float, float]],
) -> None:
    """Draw one model's curve (line + scatter)."""
    if not points:
        return
    points_sorted = sorted(points, key=lambda p: p[0])
    x = np.array([p[0] for p in points_sorted])
    y = np.array([p[1] for p in points_sorted])
    style = MODEL_STYLE.get(
        model_key,
        {"color": "#555", "marker": "o", "label": model_key.upper(), "line_color": "#555", "zorder": 0},
    )
    z = style.get("zorder", 0)
    if len(points_sorted) > 1:
        ax.plot(
            x, y,
            color=style["line_color"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            linestyle="-",
            zorder=z,
            solid_capstyle="round",
        )
    ax.scatter(
        x, y,
        c=style["color"],
        marker=style["marker"],
        s=MARKER_SIZE,
        label=style["label"],
        alpha=SCATTER_ALPHA,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_WIDTH,
        zorder=z + 10,
    )


def plot_distortion_vs_entropy(
    entropy_series: Dict[str, List[Tuple[float, float]]],
    output_dir: Path,
) -> None:
    """Plot distortion (rec loss) vs entropy (bits)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        1, 1,
        figsize=FIGURE_STYLE["figsize"],
        facecolor=FIGURE_STYLE["facecolor"],
    )
    for model_key, points in entropy_series.items():
        if points:
            _draw_curve(ax, model_key, points)
    ax.set_xlabel(
        "Entropy (bits) [log₂(perplexity)]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax.set_ylabel(
        "Distortion (reconstruction loss)",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax.set_title(
        "Distortion vs entropy",
        fontsize=AX_STYLE["title_fontsize"],
        fontweight="600",
        pad=16,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
    _style_axis(ax)
    plt.tight_layout()
    path = output_dir / "distortion_vs_entropy.png"
    plt.savefig(path, dpi=FIGURE_STYLE["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {path}")


def plot_distortion_vs_active_codes(
    codes_series: Dict[str, List[Tuple[float, float]]],
    output_dir: Path,
) -> None:
    """Plot distortion (rec loss) vs active codes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        1, 1,
        figsize=FIGURE_STYLE["figsize"],
        facecolor=FIGURE_STYLE["facecolor"],
    )
    for model_key, points in codes_series.items():
        if points:
            _draw_curve(ax, model_key, points)
    ax.set_xlabel(
        "Active codes",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax.set_ylabel(
        "Distortion (reconstruction loss)",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax.set_title(
        "Distortion vs active codes",
        fontsize=AX_STYLE["title_fontsize"],
        fontweight="600",
        pad=16,
    )
    ax.legend(loc="best", fontsize=11, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
    _style_axis(ax)
    plt.tight_layout()
    path = output_dir / "distortion_vs_active_codes.png"
    plt.savefig(path, dpi=FIGURE_STYLE["dpi"], bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot distortion vs entropy and distortion vs active codes from eval_metrics.csv",
    )
    ap.add_argument("--results-dir", type=str, default="results", help="Results root")
    ap.add_argument("--out", type=str, default="results/plots", help="Output directory for plots")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--codebook-size", type=int, default=16384, help="Prefer this codebook size when multiple exist")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return

    print("Collecting distortion / entropy / active_codes from eval_metrics.csv...")
    entropy_series, codes_series = collect_points(
        results_dir,
        image_size=args.image_size,
        prefer_codebook_size=args.codebook_size,
    )
    n_entropy = sum(len(p) for p in entropy_series.values())
    n_codes = sum(len(p) for p in codes_series.values())
    print(f"  Distortion vs entropy points: {n_entropy}")
    print(f"  Distortion vs active codes points: {n_codes}")

    if n_entropy == 0 and n_codes == 0:
        print(
            "No data found. Ensure results/<model>/<exp>/eval_metrics.csv exist with "
            "val_rec_loss, val_perplexity, val_active_codes."
        )
        return

    out_dir = Path(args.out)
    if n_entropy > 0:
        plot_distortion_vs_entropy(entropy_series, out_dir)
    if n_codes > 0:
        plot_distortion_vs_active_codes(codes_series, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()

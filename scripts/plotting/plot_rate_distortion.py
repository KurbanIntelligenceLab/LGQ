#!/usr/bin/env python3
"""
Plot rate–distortion curves for quantization models.

Rate can be expressed as:
  - log(active codes) or utilization (%) — suggestive but incomplete
  - Empirical entropy (bits) = log₂(perplexity) — marginal entropy of code assignments
  - Bits-per-token — same as entropy (bits) under i.i.d. marginal (one code per token)

Distortion: rFID, LPIPS.

Plots:
  1. rFID vs log(active codes)  — distortion vs rate (log scale)
  2. LPIPS vs utilization       — perceptual distortion vs code usage
  3. rFID vs entropy (bits)    — rate = empirical marginal entropy (stronger rate argument)
  4. LPIPS vs entropy (bits)   — same rate, perceptual distortion
  5. rFID vs bits per pixel (bpp) — principled rate for cross-method comparison
  6. LPIPS vs bpp              — perceptual distortion vs bpp
  5. Epoch labels               — (1) and (2) with epoch numbers on points (every 5th)
  6. Shaded area                — (1) and (2) with fill under each curve
  7. Small multiples            — one panel per model: epoch vs rFID (left) and LPIPS (right)

Perplexity (K_eff) is systematically included in main comparison tables; entropy (bits)
and bits-per-token are reported in evaluation and in compare_all_models_all_epochs.

Usage:
  python scripts/plot_rate_distortion.py
  python scripts/plot_rate_distortion.py --results-dir results --out results/plots
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
    "vq": {
        "color": "#8c564b",
        "marker": "o",
        "label": "VQ",
        "line_color": "#8c564b",
        "zorder": 5,
    },
    "fsq": {
        "color": "#1f77b4",
        "marker": "s",
        "label": "FSQ",
        "line_color": "#1f77b4",
        "zorder": 4,
    },
    "sim_vq": {
        "color": "#9467bd",
        "marker": "^",
        "label": "SimVQ",
        "line_color": "#9467bd",
        "zorder": 3,
    },
    "lmb": {
        "color": "#2ca02c",
        "marker": "D",
        "label": "LGQ",
        "line_color": "#2ca02c",
        "zorder": 2,
    },
    "lfq": {
        "color": "#ff7f0e",
        "marker": "p",
        "label": "LFQ",
        "line_color": "#ff7f0e",
        "zorder": 1,
    },
}

# Shared plot styling
FIGURE_STYLE = {
    "figsize": (10, 6.5),
    "facecolor": "#FAFAFA",
    "edgecolor": "#E8E8E8",
    "dpi": 200,
}
AX_STYLE = {
    "facecolor": "#FFFFFF",
    "grid_alpha": 0.35,
    "grid_linestyle": "-",
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

# LPIPS below this are treated as invalid (failed computation / missing), to avoid vertical dips to 0
LPIPS_MIN_VALID = 0.01


def _remove_lpips_outlier_dips(
    points: List[Tuple[float, float]],
    epochs: Optional[List[int]] = None,
    min_lpips: float = 0.05,
) -> Tuple[List[Tuple[float, float]], Optional[List[int]]]:
    """
    Remove spurious local minima (outlier dips) from (rate, lpips) points when sorted by rate.
    Keeps first and last point; drops a point if it is a local minimum and lpips < min_lpips.
    If epochs is provided (same length as points), returns (filtered_points, filtered_epochs).
    """
    if len(points) < 3:
        return (points, epochs if epochs is not None else None)
    # Sort by rate; keep epoch order in sync
    if epochs is not None and len(epochs) == len(points):
        combined = sorted(zip(points, epochs), key=lambda t: t[0][0])
        sorted_pts = [t[0] for t in combined]
        sorted_epochs = [t[1] for t in combined]
    else:
        combined = sorted(zip(points, range(len(points))), key=lambda t: t[0][0])
        sorted_pts = [t[0] for t in combined]
        sorted_epochs = None if epochs is None else [epochs[t[1]] for t in combined]
    keep = [True] * len(sorted_pts)
    for i in range(1, len(sorted_pts) - 1):
        lpips_val = sorted_pts[i][1]
        prev_lpips = sorted_pts[i - 1][1]
        next_lpips = sorted_pts[i + 1][1]
        if lpips_val < prev_lpips and lpips_val < next_lpips and lpips_val < min_lpips:
            keep[i] = False
    out_pts = [p for p, k in zip(sorted_pts, keep) if k]
    out_ep = [e for e, k in zip(sorted_epochs, keep) if k] if sorted_epochs is not None else None
    return (out_pts, out_ep)


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
            # per-channel: effective size is num_bins^C, often reported as 16k
            return num_bins
    except Exception:
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
    # Dedupe by cb_size: keep one with most epochs
    by_cb: Dict[int, Tuple[Path, int, int]] = {}
    for exp_dir, cb, n_epochs in candidates:
        if cb not in by_cb or n_epochs > by_cb[cb][2]:
            by_cb[cb] = (exp_dir, cb, n_epochs)
    return [(exp_dir, cb) for exp_dir, cb, _ in by_cb.values()]


def collect_rate_distortion_points(
    results_dir: Path,
    image_size: int = 128,
    prefer_codebook_size: int = 16384,
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Collect (rate, distortion) points per model.
    Returns dict: model_key -> [(rate, distortion), ...] for each epoch.
    """
    models = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
    data: Dict[str, List] = {m: [] for m in models}
    rfid_epochs: Dict[str, List[int]] = {m: [] for m in models}
    lpips_epochs: Dict[str, List[int]] = {m: [] for m in models}
    epoch_metrics: Dict[str, List[Tuple[int, float, float]]] = {m: [] for m in models}
    perplexity_epoch: Dict[str, List[Tuple[int, float]]] = {m: [] for m in models}
    entropy_epoch: Dict[str, List[Tuple[int, float]]] = {m: [] for m in models}

    for model_key in models:
        experiments = find_experiments(
            results_dir, model_key, image_size=image_size
        )
        chosen = None
        for exp_dir, cb in experiments:
            if cb == prefer_codebook_size:
                chosen = (exp_dir, cb)
                break
        if chosen is None and experiments:
            chosen = max(experiments, key=lambda x: x[1])
        if chosen is None:
            continue
        exp_dir, cb_size = chosen
        epochs_metrics = load_epoch_metrics(exp_dir)
        if not epochs_metrics:
            continue
        for epoch, m in sorted(epochs_metrics.items()):
            rfid = m.get("val_rfid")
            lpips = m.get("val_lpips")
            active = m.get("val_active_codes")
            util = m.get("val_codebook_util")
            ppl = m.get("val_perplexity")
            entropy_bits = m.get("val_entropy_bits")
            if entropy_bits is None and ppl is not None and ppl > 0:
                entropy_bits = math.log2(float(ppl))
            # Fix wrong 0/None codebook_util (e.g. VQ CSV sometimes has 0.0 despite valid active_codes)
            if (util is None or (isinstance(util, (int, float)) and util == 0)) and active is not None and cb_size and cb_size > 0:
                util = 100.0 * active / cb_size
            rate_log = math.log10(active) if (active is not None and active > 0) else None
            # Bits per pixel: from val_bpp or entropy/256 (backbone 16× downsample)
            bpp = m.get("val_bpp")
            if bpp is None and entropy_bits is not None and entropy_bits > 0:
                bpp = entropy_bits / (16.0 ** 2)
            if rfid is not None and rate_log is not None:
                data[model_key].append(("rfid", rate_log, rfid))
                rfid_epochs[model_key].append(epoch)
            if lpips is not None and lpips >= LPIPS_MIN_VALID and util is not None:
                data[model_key].append(("lpips_util", util, lpips))
                lpips_epochs[model_key].append(epoch)
            if rfid is not None and lpips is not None and lpips >= LPIPS_MIN_VALID and rate_log is not None:
                data[model_key].append(("rfid_lpips", rfid, lpips, rate_log))
            if rfid is not None and lpips is not None and lpips >= LPIPS_MIN_VALID:
                epoch_metrics[model_key].append((epoch, rfid, lpips))
            # Rate = empirical entropy (bits) for stronger rate argument
            if rfid is not None and entropy_bits is not None:
                data[model_key].append(("rfid_entropy", entropy_bits, rfid))
            if lpips is not None and lpips >= LPIPS_MIN_VALID and entropy_bits is not None:
                data[model_key].append(("lpips_entropy", entropy_bits, lpips))
            # Rate = bits per pixel (principled rate for cross-method comparison)
            if rfid is not None and bpp is not None and bpp > 0:
                data[model_key].append(("rfid_bpp", bpp, rfid))
            if lpips is not None and lpips >= LPIPS_MIN_VALID and bpp is not None and bpp > 0:
                data[model_key].append(("lpips_bpp", bpp, lpips))
            # Perplexity / entropy vs epoch curves
            if ppl is not None and ppl > 0:
                perplexity_epoch[model_key].append((epoch, float(ppl)))
            if entropy_bits is not None:
                entropy_epoch[model_key].append((epoch, float(entropy_bits)))

    rfid_series = {k: [(x[1], x[2]) for x in v if x[0] == "rfid"] for k, v in data.items()}
    lpips_series = {k: [(x[1], x[2]) for x in v if x[0] == "lpips_util"] for k, v in data.items()}
    rfid_entropy_series = {k: [(x[1], x[2]) for x in v if x[0] == "rfid_entropy"] for k, v in data.items()}
    lpips_entropy_series = {k: [(x[1], x[2]) for x in v if x[0] == "lpips_entropy"] for k, v in data.items()}
    rfid_bpp_series = {k: [(x[1], x[2]) for x in v if x[0] == "rfid_bpp"] for k, v in data.items()}
    lpips_bpp_series = {k: [(x[1], x[2]) for x in v if x[0] == "lpips_bpp"] for k, v in data.items()}
    # Remove spurious LPIPS dips (local minima < 0.05 when sorted by rate); keep lpips_epochs in sync
    for k in list(lpips_series.keys()):
        pts, ep = _remove_lpips_outlier_dips(lpips_series[k], lpips_epochs.get(k), min_lpips=0.05)
        lpips_series[k] = pts
        if ep is not None:
            lpips_epochs[k] = ep
    for k in list(lpips_entropy_series.keys()):
        lpips_entropy_series[k], _ = _remove_lpips_outlier_dips(
            lpips_entropy_series[k], None, min_lpips=0.05
        )
    for k in list(lpips_bpp_series.keys()):
        lpips_bpp_series[k], _ = _remove_lpips_outlier_dips(
            lpips_bpp_series[k], None, min_lpips=0.05
        )
    rfid_lpips_series = {
        k: [(x[1], x[2], x[3]) for x in v if x[0] == "rfid_lpips"]
        for k, v in data.items()
    }
    return {
        "rfid": rfid_series,
        "lpips": lpips_series,
        "rfid_entropy": rfid_entropy_series,
        "lpips_entropy": lpips_entropy_series,
        "rfid_bpp": rfid_bpp_series,
        "lpips_bpp": lpips_bpp_series,
        "rfid_lpips": rfid_lpips_series,
        "rfid_epochs": rfid_epochs,
        "lpips_epochs": lpips_epochs,
        "epoch_metrics": epoch_metrics,
        "perplexity_epoch": perplexity_epoch,
        "entropy_epoch": entropy_epoch,
    }


def _style_axis(ax: plt.Axes) -> None:
    """Apply consistent axis styling (grid, spines, fonts)."""
    ax.set_facecolor(AX_STYLE["facecolor"])
    ax.grid(
        True,
        alpha=AX_STYLE["grid_alpha"],
        linestyle=AX_STYLE["grid_linestyle"],
        linewidth=0.8,
    )
    ax.tick_params(axis="both", labelsize=AX_STYLE["tick_fontsize"])
    for spine in ax.spines.values():
        spine.set_color(AX_STYLE["spine_color"])
        spine.set_linewidth(AX_STYLE["spine_lw"])


def _draw_one_curve(
    ax: plt.Axes,
    model_key: str,
    points: List[Tuple[float, float]],
) -> None:
    """Draw a single model's rate-distortion curve (line with markers at start and end only)."""
    points_sorted = sorted(points, key=lambda p: p[0])
    x = np.array([p[0] for p in points_sorted])
    y = np.array([p[1] for p in points_sorted])
    style = MODEL_STYLE.get(
        model_key,
        {"color": "#555555", "marker": "o", "label": model_key.upper(), "line_color": "#555555", "zorder": 0},
    )
    z = style.get("zorder", 0)
    if len(points_sorted) > 1:
        ax.plot(
            x,
            y,
            color=style["line_color"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            linestyle="-",
            zorder=z,
            solid_capstyle="round",
            label=style["label"],
        )
        ax.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            c=style["color"],
            marker=style["marker"],
            s=MARKER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            zorder=z + 10,
        )
    else:
        ax.scatter(
            x,
            y,
            c=style["color"],
            marker=style["marker"],
            s=MARKER_SIZE,
            label=style["label"],
            alpha=SCATTER_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            zorder=z + 10,
        )


def _draw_one_curve_with_fill(
    ax: plt.Axes,
    model_key: str,
    points: List[Tuple[float, float]],
) -> None:
    """Draw rate-distortion curve with shaded area; markers at start and end only."""
    points_sorted = sorted(points, key=lambda p: p[0])
    x = np.array([p[0] for p in points_sorted])
    y = np.array([p[1] for p in points_sorted])
    style = MODEL_STYLE.get(
        model_key,
        {"color": "#555555", "marker": "o", "label": model_key.upper(), "line_color": "#555555", "zorder": 0},
    )
    z = style.get("zorder", 0)
    y_baseline = 0.0
    ax.fill_between(
        x,
        y_baseline,
        y,
        color=style["line_color"],
        alpha=0.18,
        zorder=z,
    )
    if len(points_sorted) > 1:
        ax.plot(
            x,
            y,
            color=style["line_color"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            linestyle="-",
            zorder=z + 1,
            solid_capstyle="round",
            label=style["label"],
        )
        ax.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            c=style["color"],
            marker=style["marker"],
            s=MARKER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            zorder=z + 10,
        )
    else:
        ax.scatter(
            x,
            y,
            c=style["color"],
            marker=style["marker"],
            s=MARKER_SIZE,
            label=style["label"],
            alpha=SCATTER_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            zorder=z + 10,
        )


def _draw_one_curve_with_epoch_labels(
    ax: plt.Axes,
    model_key: str,
    points: List[Tuple[float, float]],
    epochs: List[int],
    every_n: int = 5,
) -> None:
    """Draw rate-distortion curve with markers and epoch labels at start and end only."""
    if len(epochs) != len(points):
        _draw_one_curve(ax, model_key, points)
        return
    combined = sorted(zip(points, epochs), key=lambda t: t[0][0])
    points_sorted = [t[0] for t in combined]
    epochs_sorted = [t[1] for t in combined]
    x = np.array([p[0] for p in points_sorted])
    y = np.array([p[1] for p in points_sorted])
    style = MODEL_STYLE.get(
        model_key,
        {"color": "#555555", "marker": "o", "label": model_key.upper(), "line_color": "#555555", "zorder": 0},
    )
    z = style.get("zorder", 0)
    if len(points_sorted) > 1:
        ax.plot(
            x,
            y,
            color=style["line_color"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            linestyle="-",
            zorder=z,
            solid_capstyle="round",
            label=style["label"],
        )
        ax.scatter(
            [x[0], x[-1]],
            [y[0], y[-1]],
            c=style["color"],
            marker=style["marker"],
            s=MARKER_SIZE,
            alpha=SCATTER_ALPHA,
            edgecolors=EDGE_COLOR,
            linewidths=EDGE_WIDTH,
            zorder=z + 10,
        )
        for i in (0, -1):
            ax.annotate(
                str(epochs_sorted[i]),
                (x[i], y[i]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=style["line_color"],
                alpha=0.9,
            )


def _draw_one_curve_line_with_dot_at_epoch(
    ax: plt.Axes,
    model_key: str,
    points: List[Tuple[float, float]],
    epochs: List[int],
    at_epoch: int,
) -> None:
    """Draw rate-distortion curve sorted by rate (smooth line) plus prominent dot at at_epoch."""
    if len(epochs) != len(points):
        return
    # Restrict to epoch <= at_epoch
    filtered = [(p, e) for p, e in zip(points, epochs) if e <= at_epoch]
    if not filtered:
        return
    # (xi, yi) for the dot at at_epoch (actual rate, distortion at that epoch)
    idx_dot = min(range(len(filtered)), key=lambda i: abs(filtered[i][1] - at_epoch))
    xi, yi = filtered[idx_dot][0][0], filtered[idx_dot][0][1]
    epoch_label = filtered[idx_dot][1]
    # Line: sort by rate (x) so curve is smooth left-to-right, not zigzag by epoch
    sorted_by_rate = sorted(filtered, key=lambda t: t[0][0])
    x = np.array([t[0][0] for t in sorted_by_rate])
    y = np.array([t[0][1] for t in sorted_by_rate])
    style = MODEL_STYLE.get(
        model_key,
        {"color": "#555555", "marker": "o", "label": model_key.upper(), "line_color": "#555555", "zorder": 0},
    )
    z = style.get("zorder", 0)
    if len(x) > 1:
        ax.plot(
            x,
            y,
            color=style["line_color"],
            linewidth=LINE_WIDTH,
            alpha=LINE_ALPHA,
            linestyle="-",
            zorder=z,
            solid_capstyle="round",
            label=style["label"],
        )
    elif len(x) == 1:
        ax.scatter(x, y, c=style["color"], marker="o", s=MARKER_SIZE, alpha=LINE_ALPHA, zorder=z, label=style["label"])
    # Same small circle for all models at epoch label
    ax.scatter(
        [xi],
        [yi],
        c=style["color"],
        marker="o",
        s=MARKER_SIZE,
        alpha=1.0,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_WIDTH,
        zorder=z + 10,
    )
    ax.annotate(
        str(epoch_label),
        (xi, yi),
        xytext=(6, 6),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        color=style["line_color"],
        alpha=0.95,
    )


def _draw_one_curve_dots_at_epoch(
    ax: plt.Axes,
    model_key: str,
    points: List[Tuple[float, float]],
    epochs: List[int],
    at_epoch: int,
) -> None:
    """Draw only the dot at at_epoch (no lines, no other dots)."""
    if len(epochs) != len(points):
        return
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    epochs_list = list(epochs)
    style = MODEL_STYLE.get(
        model_key,
        {"color": "#555555", "marker": "o", "label": model_key.upper(), "line_color": "#555555", "zorder": 0},
    )
    z = style.get("zorder", 0)
    idx = min(range(len(epochs_list)), key=lambda i: abs(epochs_list[i] - at_epoch))
    xi, yi = x[idx], y[idx]
    ax.scatter(
        [xi],
        [yi],
        c=style["color"],
        marker=style["marker"],
        s=MARKER_SIZE * 3.0,
        alpha=1.0,
        edgecolors=EDGE_COLOR,
        linewidths=EDGE_WIDTH * 1.5,
        zorder=z + 10,
        label=style["label"],
    )


def plot_rate_distortion(
    series: Dict[str, Dict[str, List[Tuple[float, float]]]],
    output_dir: Path,
    max_epoch: Optional[int] = None,
) -> None:
    """Produce two figures: rFID vs log(active codes), LPIPS vs utilization."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) rFID vs log(active codes)
    fig1, ax1 = plt.subplots(
        1,
        1,
        figsize=FIGURE_STYLE["figsize"],
        facecolor=FIGURE_STYLE["facecolor"],
    )
    for model_key, points in series["rfid"].items():
        if not points:
            continue
        _draw_one_curve(ax1, model_key, points)
    ax1.set_xlabel(
        "log₁₀(active codes) [rate]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax1.set_ylabel(
        "rFID [distortion]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax1.set_title(
        "Rate–Distortion: rFID vs log(active codes)",
        fontsize=AX_STYLE["title_fontsize"],
        fontweight="600",
        pad=16,
    )
    ax1.legend(
        loc="best",
        fontsize=11,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fancybox=False,
    )
    _style_axis(ax1)
    plt.tight_layout()
    p1 = output_dir / "rate_distortion_rfid_vs_log_active_codes.png"
    plt.savefig(
        p1,
        dpi=FIGURE_STYLE["dpi"],
        bbox_inches="tight",
        facecolor=fig1.get_facecolor(),
        edgecolor="none",
    )
    plt.close()
    print(f"Saved: {p1}")

    # 2) LPIPS vs utilization
    fig2, ax2 = plt.subplots(
        1,
        1,
        figsize=FIGURE_STYLE["figsize"],
        facecolor=FIGURE_STYLE["facecolor"],
    )
    for model_key, points in series["lpips"].items():
        if not points:
            continue
        _draw_one_curve(ax2, model_key, points)
    ax2.set_xlabel(
        "Codebook utilization (%) [rate]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax2.set_ylabel(
        "LPIPS [distortion]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax2.set_title(
        "Rate–Distortion: LPIPS vs utilization",
        fontsize=AX_STYLE["title_fontsize"],
        fontweight="600",
        pad=16,
    )
    ax2.legend(
        loc="best",
        fontsize=11,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fancybox=False,
    )
    _style_axis(ax2)
    plt.tight_layout()
    p2 = output_dir / "rate_distortion_lpips_vs_utilization.png"
    plt.savefig(
        p2,
        dpi=FIGURE_STYLE["dpi"],
        bbox_inches="tight",
        facecolor=fig2.get_facecolor(),
        edgecolor="none",
    )
    plt.close()
    print(f"Saved: {p2}")

    # 2c) rFID vs entropy (bits) — rate = empirical marginal entropy
    rfid_entropy = series.get("rfid_entropy", {})
    if any(rfid_entropy.values()):
        fig_entropy_rfid, ax_er = plt.subplots(
            1, 1, figsize=FIGURE_STYLE["figsize"], facecolor=FIGURE_STYLE["facecolor"]
        )
        for model_key, points in rfid_entropy.items():
            if not points:
                continue
            _draw_one_curve(ax_er, model_key, points)
        ax_er.set_xlabel(
            "Entropy (bits) [log₂(perplexity)]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_er.set_ylabel(
            "rFID [distortion]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_er.set_title(
            "Rate–Distortion: rFID vs entropy (bits)",
            fontsize=AX_STYLE["title_fontsize"],
            fontweight="600",
            pad=16,
        )
        ax_er.legend(
            loc="best",
            fontsize=11,
            framealpha=0.95,
            edgecolor="#DDDDDD",
            fancybox=False,
        )
        _style_axis(ax_er)
        plt.tight_layout()
        p_entropy_rfid = output_dir / "rate_distortion_rfid_vs_entropy_bits.png"
        plt.savefig(
            p_entropy_rfid,
            dpi=FIGURE_STYLE["dpi"],
            bbox_inches="tight",
            facecolor=fig_entropy_rfid.get_facecolor(),
            edgecolor="none",
        )
        plt.close()
        print(f"Saved: {p_entropy_rfid}")

    # 2d) LPIPS vs entropy (bits)
    lpips_entropy = series.get("lpips_entropy", {})
    if any(lpips_entropy.values()):
        fig_entropy_lpips, ax_el = plt.subplots(
            1, 1, figsize=FIGURE_STYLE["figsize"], facecolor=FIGURE_STYLE["facecolor"]
        )
        for model_key, points in lpips_entropy.items():
            if not points:
                continue
            _draw_one_curve(ax_el, model_key, points)
        ax_el.set_xlabel(
            "Entropy (bits) [log₂(perplexity)]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_el.set_ylabel(
            "LPIPS [distortion]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_el.set_title(
            "Rate–Distortion: LPIPS vs entropy (bits)",
            fontsize=AX_STYLE["title_fontsize"],
            fontweight="600",
            pad=16,
        )
        ax_el.legend(
            loc="best",
            fontsize=11,
            framealpha=0.95,
            edgecolor="#DDDDDD",
            fancybox=False,
        )
        _style_axis(ax_el)
        plt.tight_layout()
        p_entropy_lpips = output_dir / "rate_distortion_lpips_vs_entropy_bits.png"
        plt.savefig(
            p_entropy_lpips,
            dpi=FIGURE_STYLE["dpi"],
            bbox_inches="tight",
            facecolor=fig_entropy_lpips.get_facecolor(),
            edgecolor="none",
        )
        plt.close()
        print(f"Saved: {p_entropy_lpips}")

    # 2e) rFID vs bits per pixel (bpp) — principled rate for cross-method comparison
    rfid_bpp = series.get("rfid_bpp", {})
    if any(rfid_bpp.values()):
        fig_bpp_rfid, ax_br = plt.subplots(
            1, 1, figsize=FIGURE_STYLE["figsize"], facecolor=FIGURE_STYLE["facecolor"]
        )
        for model_key, points in rfid_bpp.items():
            if not points:
                continue
            _draw_one_curve(ax_br, model_key, points)
        ax_br.set_xlabel(
            "Bits per pixel (bpp) [rate]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_br.set_ylabel(
            "rFID [distortion]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_br.set_title(
            "Rate–Distortion: rFID vs bits per pixel",
            fontsize=AX_STYLE["title_fontsize"],
            fontweight="600",
            pad=16,
        )
        ax_br.legend(
            loc="best",
            fontsize=11,
            framealpha=0.95,
            edgecolor="#DDDDDD",
            fancybox=False,
        )
        _style_axis(ax_br)
        plt.tight_layout()
        p_bpp_rfid = output_dir / "rate_distortion_rfid_vs_bpp.png"
        plt.savefig(
            p_bpp_rfid,
            dpi=FIGURE_STYLE["dpi"],
            bbox_inches="tight",
            facecolor=fig_bpp_rfid.get_facecolor(),
            edgecolor="none",
        )
        plt.close()
        print(f"Saved: {p_bpp_rfid}")

    # 2f) LPIPS vs bits per pixel (bpp)
    lpips_bpp = series.get("lpips_bpp", {})
    if any(lpips_bpp.values()):
        fig_bpp_lpips, ax_bl = plt.subplots(
            1, 1, figsize=FIGURE_STYLE["figsize"], facecolor=FIGURE_STYLE["facecolor"]
        )
        for model_key, points in lpips_bpp.items():
            if not points:
                continue
            _draw_one_curve(ax_bl, model_key, points)
        ax_bl.set_xlabel(
            "Bits per pixel (bpp) [rate]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_bl.set_ylabel(
            "LPIPS [distortion]",
            fontsize=AX_STYLE["label_fontsize"],
            fontweight="600",
        )
        ax_bl.set_title(
            "Rate–Distortion: LPIPS vs bits per pixel",
            fontsize=AX_STYLE["title_fontsize"],
            fontweight="600",
            pad=16,
        )
        ax_bl.legend(
            loc="best",
            fontsize=11,
            framealpha=0.95,
            edgecolor="#DDDDDD",
            fancybox=False,
        )
        _style_axis(ax_bl)
        plt.tight_layout()
        p_bpp_lpips = output_dir / "rate_distortion_lpips_vs_bpp.png"
        plt.savefig(
            p_bpp_lpips,
            dpi=FIGURE_STYLE["dpi"],
            bbox_inches="tight",
            facecolor=fig_bpp_lpips.get_facecolor(),
            edgecolor="none",
        )
        plt.close()
        print(f"Saved: {p_bpp_lpips}")

    # 2b) Combined: rFID and LPIPS in one image (top and bottom)
    fig_combined, (ax_combined_a, ax_combined_b) = plt.subplots(
        2,
        1,
        figsize=(10, 12),
        facecolor=FIGURE_STYLE["facecolor"],
        gridspec_kw={"hspace": 0.32},
    )
    for model_key, points in series["rfid"].items():
        if not points:
            continue
        _draw_one_curve(ax_combined_a, model_key, points)
    ax_combined_a.set_xlabel(
        "log₁₀(active codes) [rate]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax_combined_a.set_ylabel(
        "rFID [distortion]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax_combined_a.set_title(
        "rFID vs log(active codes)",
        fontsize=AX_STYLE["title_fontsize"],
        fontweight="600",
        pad=12,
    )
    ax_combined_a.legend(
        loc="best",
        fontsize=11,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fancybox=False,
    )
    _style_axis(ax_combined_a)
    for model_key, points in series["lpips"].items():
        if not points:
            continue
        _draw_one_curve(ax_combined_b, model_key, points)
    ax_combined_b.set_xlabel(
        "Codebook utilization (%) [rate]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax_combined_b.set_ylabel(
        "LPIPS [distortion]",
        fontsize=AX_STYLE["label_fontsize"],
        fontweight="600",
    )
    ax_combined_b.set_title(
        "LPIPS vs utilization",
        fontsize=AX_STYLE["title_fontsize"],
        fontweight="600",
        pad=12,
    )
    ax_combined_b.legend(
        loc="best",
        fontsize=11,
        framealpha=0.95,
        edgecolor="#DDDDDD",
        fancybox=False,
    )
    _style_axis(ax_combined_b)
    plt.suptitle(
        "Rate–Distortion",
        fontsize=16,
        fontweight="600",
        y=1.01,
    )
    plt.tight_layout()
    p_combined = output_dir / "rate_distortion_rfid_and_lpips.png"
    plt.savefig(
        p_combined,
        dpi=FIGURE_STYLE["dpi"],
        bbox_inches="tight",
        facecolor=fig_combined.get_facecolor(),
        edgecolor="none",
    )
    plt.close()
    print(f"Saved: {p_combined}")

    # 3) Epoch labels on curves (top: rFID, bottom: LPIPS; optional max_epoch filter)
    rfid_epochs = series.get("rfid_epochs", {})
    lpips_epochs = series.get("lpips_epochs", {})
    if max_epoch is not None:
        rfid_f = {
            k: [(p[0], p[1]) for p, e in zip(series["rfid"][k], rfid_epochs.get(k, [])) if e <= max_epoch]
            for k in series["rfid"]
        }
        rfid_ep_f = {
            k: [e for e in rfid_epochs.get(k, []) if e <= max_epoch]
            for k in series["rfid"]
        }
        lpips_f = {
            k: [(p[0], p[1]) for p, e in zip(series["lpips"][k], lpips_epochs.get(k, [])) if e <= max_epoch]
            for k in series["lpips"]
        }
        lpips_ep_f = {
            k: [e for e in lpips_epochs.get(k, []) if e <= max_epoch]
            for k in series["lpips"]
        }
    else:
        rfid_f = series["rfid"]
        rfid_ep_f = rfid_epochs
        lpips_f = series["lpips"]
        lpips_ep_f = lpips_epochs
    fig3, (ax3a, ax3b) = plt.subplots(
        2, 1, figsize=(8, 10), facecolor=FIGURE_STYLE["facecolor"], gridspec_kw={"hspace": 0.32}
    )
    # Draw full rate-distortion lines (up to max_epoch) with prominent dot + label at epoch 60
    highlight_epoch = max_epoch if max_epoch is not None else 60
    for model_key, points in rfid_f.items():
        if not points:
            continue
        epochs = rfid_ep_f.get(model_key, [])
        _draw_one_curve_line_with_dot_at_epoch(ax3a, model_key, points, epochs, at_epoch=highlight_epoch)
    ax3a.set_xlabel("log₁₀(active codes) [rate]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax3a.set_ylabel("rFID [distortion]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax3a.set_title(f"rFID vs rate (epoch labels)", fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=12)
    _style_axis(ax3a)
    for model_key, points in lpips_f.items():
        if not points:
            continue
        epochs = lpips_ep_f.get(model_key, [])
        _draw_one_curve_line_with_dot_at_epoch(ax3b, model_key, points, epochs, at_epoch=highlight_epoch)
    ax3b.set_xlabel("Codebook utilization (%) [rate]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax3b.set_ylabel("LPIPS [distortion]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax3b.set_title(f"LPIPS vs rate (epoch labels)", fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=12)
    ax3b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, fontsize=10, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
    _style_axis(ax3b)
    suptitle = "Rate–Distortion (lines up to epoch " + (str(max_epoch) if max_epoch is not None else "60") + ")"
    plt.suptitle(suptitle, fontsize=16, fontweight="600", y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    p3 = output_dir / "rate_distortion_epoch_labels.png"
    plt.savefig(p3, dpi=FIGURE_STYLE["dpi"], bbox_inches="tight", facecolor=fig3.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {p3}")

    # 4) Shaded area under each curve
    fig4, (ax4a, ax4b) = plt.subplots(
        1, 2, figsize=(14, 6), facecolor=FIGURE_STYLE["facecolor"], gridspec_kw={"wspace": 0.28}
    )
    for model_key, points in series["rfid"].items():
        if points:
            _draw_one_curve_with_fill(ax4a, model_key, points)
    ax4a.set_xlabel("log₁₀(active codes) [rate]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax4a.set_ylabel("rFID [distortion]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax4a.set_title("rFID vs rate (shaded)", fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=12)
    ax4a.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
    _style_axis(ax4a)
    for model_key, points in series["lpips"].items():
        if points:
            _draw_one_curve_with_fill(ax4b, model_key, points)
    ax4b.set_xlabel("Codebook utilization (%) [rate]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax4b.set_ylabel("LPIPS [distortion]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
    ax4b.set_title("LPIPS vs rate (shaded)", fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=12)
    ax4b.legend(loc="best", fontsize=10, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
    _style_axis(ax4b)
    plt.suptitle("Rate–Distortion (area under curve)", fontsize=16, fontweight="600", y=1.02)
    plt.tight_layout()
    p4 = output_dir / "rate_distortion_shaded.png"
    plt.savefig(p4, dpi=FIGURE_STYLE["dpi"], bbox_inches="tight", facecolor=fig4.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Saved: {p4}")

    # 5) Small multiples: one panel per model, epoch vs rFID (left) and LPIPS (right)
    epoch_metrics = series.get("epoch_metrics", {})
    if epoch_metrics and any(epoch_metrics.values()):
        n_models = sum(1 for v in epoch_metrics.values() if v)
        if n_models > 0:
            ncols = 3
            nrows = (n_models + ncols - 1) // ncols
            fig5, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(5 * ncols, 4 * nrows),
                facecolor=FIGURE_STYLE["facecolor"],
                squeeze=False,
            )
            axes_flat = axes.flatten()
            for idx, (model_key, em_list) in enumerate(epoch_metrics.items()):
                if not em_list or idx >= len(axes_flat):
                    continue
                ax_left = axes_flat[idx]
                epochs_arr = np.array([t[0] for t in em_list])
                rfid_arr = np.array([t[1] for t in em_list])
                lpips_arr = np.array([t[2] for t in em_list])
                style = MODEL_STYLE.get(
                    model_key,
                    {"color": "#555555", "line_color": "#555555", "label": model_key.upper()},
                )
                ax_left.plot(
                    epochs_arr,
                    rfid_arr,
                    color=style["line_color"],
                    linewidth=2,
                    alpha=0.9,
                    label="rFID",
                )
                ax_left.set_xlabel("Epoch", fontsize=AX_STYLE["tick_fontsize"], fontweight="600")
                ax_left.set_ylabel("rFID", fontsize=AX_STYLE["tick_fontsize"], fontweight="600", color=style["line_color"])
                ax_left.tick_params(axis="y", labelcolor=style["line_color"])
                ax_left.set_facecolor(AX_STYLE["facecolor"])
                ax_right = ax_left.twinx()
                ax_right.plot(
                    epochs_arr,
                    lpips_arr,
                    color="#666666",
                    linewidth=2,
                    alpha=0.7,
                    linestyle="--",
                    label="LPIPS",
                )
                ax_right.set_ylabel("LPIPS", fontsize=AX_STYLE["tick_fontsize"], fontweight="600", color="#666666")
                ax_right.tick_params(axis="y", labelcolor="#666666")
                ax_left.set_title(style["label"], fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=8)
                _style_axis(ax_left)
            for j in range(len(epoch_metrics), len(axes_flat)):
                axes_flat[j].set_visible(False)
            plt.suptitle("Per-model: rFID and LPIPS over training", fontsize=16, fontweight="600", y=1.01)
            fig5.subplots_adjust(hspace=0.35, wspace=0.3)
            p5 = output_dir / "rate_distortion_small_multiples.png"
            plt.savefig(p5, dpi=FIGURE_STYLE["dpi"], bbox_inches="tight", facecolor=fig5.get_facecolor(), edgecolor="none")
            plt.close()
            print(f"Saved: {p5}")

    # 6) Perplexity vs epoch and Entropy (bits) vs epoch
    perplexity_epoch = series.get("perplexity_epoch", {})
    entropy_epoch = series.get("entropy_epoch", {})
    if any(perplexity_epoch.values()) or any(entropy_epoch.values()):
        fig6, (ax6a, ax6b) = plt.subplots(
            1, 2, figsize=(14, 6), facecolor=FIGURE_STYLE["facecolor"], gridspec_kw={"wspace": 0.28}
        )
        for model_key, points in perplexity_epoch.items():
            if not points:
                continue
            points_sorted = sorted(points, key=lambda p: p[0])
            epochs_arr = np.array([p[0] for p in points_sorted])
            ppl_arr = np.array([p[1] for p in points_sorted])
            style = MODEL_STYLE.get(
                model_key,
                {"color": "#555555", "line_color": "#555555", "label": model_key.upper(), "marker": "o"},
            )
            ax6a.plot(
                epochs_arr,
                ppl_arr,
                color=style["line_color"],
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
                linestyle="-",
                zorder=style.get("zorder", 0),
                label=style["label"],
            )
            if len(epochs_arr) > 1:
                ax6a.scatter(
                    [epochs_arr[0], epochs_arr[-1]],
                    [ppl_arr[0], ppl_arr[-1]],
                    c=style["color"],
                    marker=style["marker"],
                    s=MARKER_SIZE,
                    alpha=SCATTER_ALPHA,
                    edgecolors=EDGE_COLOR,
                    linewidths=EDGE_WIDTH,
                )
            else:
                ax6a.scatter(
                    epochs_arr,
                    ppl_arr,
                    c=style["color"],
                    marker=style["marker"],
                    s=MARKER_SIZE,
                    label=style["label"],
                    alpha=SCATTER_ALPHA,
                    edgecolors=EDGE_COLOR,
                    linewidths=EDGE_WIDTH,
                )
        ax6a.set_xlabel("Epoch", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
        ax6a.set_ylabel("Perplexity ($K_{\\mathrm{eff}}$)", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
        ax6a.set_title("Perplexity vs epoch", fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=12)
        ax6a.legend(loc="best", fontsize=11, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
        _style_axis(ax6a)
        for model_key, points in entropy_epoch.items():
            if not points:
                continue
            points_sorted = sorted(points, key=lambda p: p[0])
            epochs_arr = np.array([p[0] for p in points_sorted])
            entropy_arr = np.array([p[1] for p in points_sorted])
            style = MODEL_STYLE.get(
                model_key,
                {"color": "#555555", "line_color": "#555555", "label": model_key.upper(), "marker": "o"},
            )
            ax6b.plot(
                epochs_arr,
                entropy_arr,
                color=style["line_color"],
                linewidth=LINE_WIDTH,
                alpha=LINE_ALPHA,
                linestyle="-",
                zorder=style.get("zorder", 0),
                label=style["label"],
            )
            if len(epochs_arr) > 1:
                ax6b.scatter(
                    [epochs_arr[0], epochs_arr[-1]],
                    [entropy_arr[0], entropy_arr[-1]],
                    c=style["color"],
                    marker=style["marker"],
                    s=MARKER_SIZE,
                    alpha=SCATTER_ALPHA,
                    edgecolors=EDGE_COLOR,
                    linewidths=EDGE_WIDTH,
                )
            else:
                ax6b.scatter(
                    epochs_arr,
                    entropy_arr,
                    c=style["color"],
                    marker=style["marker"],
                    s=MARKER_SIZE,
                    label=style["label"],
                    alpha=SCATTER_ALPHA,
                    edgecolors=EDGE_COLOR,
                    linewidths=EDGE_WIDTH,
                )
        ax6b.set_xlabel("Epoch", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
        ax6b.set_ylabel("Entropy (bits) [log₂(perplexity)]", fontsize=AX_STYLE["label_fontsize"], fontweight="600")
        ax6b.set_title("Marginal entropy vs epoch", fontsize=AX_STYLE["title_fontsize"], fontweight="600", pad=12)
        ax6b.legend(loc="best", fontsize=11, framealpha=0.95, edgecolor="#DDDDDD", fancybox=False)
        _style_axis(ax6b)
        plt.suptitle("Perplexity and marginal entropy over training", fontsize=16, fontweight="600", y=1.02)
        plt.tight_layout()
        p6 = output_dir / "rate_distortion_perplexity_entropy_vs_epoch.png"
        plt.savefig(p6, dpi=FIGURE_STYLE["dpi"], bbox_inches="tight", facecolor=fig6.get_facecolor(), edgecolor="none")
        plt.close()
        print(f"Saved: {p6}")


def main():
    ap = argparse.ArgumentParser(description="Plot rate–distortion curves (rFID vs log(active codes), LPIPS vs utilization)")
    ap.add_argument("--results-dir", type=str, default="results", help="Results root (e.g. results)")
    ap.add_argument("--out", type=str, default="results/plots", help="Output directory for plots")
    ap.add_argument("--image-size", type=int, default=128)
    ap.add_argument("--codebook-size", type=int, default=16384, help="Prefer this codebook size when multiple exist")
    ap.add_argument("--max-epoch", type=int, default=41, help="Epoch labels figure: only show data up to this epoch (default: 41)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results dir not found: {results_dir}")
        return

    print("Collecting rate–distortion points from eval_metrics.csv...")
    series = collect_rate_distortion_points(
        results_dir,
        image_size=args.image_size,
        prefer_codebook_size=args.codebook_size,
    )
    n_rfid = sum(len(p) for p in series["rfid"].values())
    n_lpips = sum(len(p) for p in series["lpips"].values())
    print(f"  rFID points: {n_rfid}, LPIPS points: {n_lpips}")

    if n_rfid == 0 and n_lpips == 0:
        print("No data found. Ensure results/<model>/<exp>/eval_metrics.csv exist with val_rfid, val_lpips, val_active_codes, val_codebook_util.")
        return

    out_dir = Path(args.out)
    plot_rate_distortion(series, out_dir, max_epoch=args.max_epoch)
    print("Done.")


if __name__ == "__main__":
    main()

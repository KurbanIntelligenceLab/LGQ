#!/usr/bin/env python3
"""
Stability across seeds and epochs: mean ± std of rFID, active codes, perplexity (K_eff).

Plots:
  - rFID vs epoch (mean ± std over seeds)
  - Active codes vs epoch (mean ± std over seeds)
  - Perplexity / K_eff vs epoch (mean ± std over seeds)

Why: VQ collapse is unstable; forced-utilization methods look stable but rigid;
LGQ should show early stabilization with low variance.

Usage:
  # Config file: JSON mapping model name -> list of experiment dirs (one per seed)
  python scripts/plot_stability_seeds.py --config results/stability_seeds_config.json --out results/plots

  # Single run per model (std = 0): use existing find_experiment logic
  python scripts/plot_stability_seeds.py --results-dir results --out results/plots

  # Fixed-epoch table: mean ± std at epoch 5
  python scripts/plot_stability_seeds.py --config ... --fixed-epoch 5
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


def load_epoch_metrics(experiment_dir: Path) -> Dict[int, Dict[str, float]]:
    """Load metrics per epoch from eval_metrics.csv. Keeps first row per epoch."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return {}
    epochs_metrics: Dict[int, Dict[str, float]] = {}
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
                        if value and str(value).strip() and str(value) != "None":
                            v = float(value)
                            if not (math.isnan(v) or math.isinf(v)):
                                metrics[key] = v
                    except (ValueError, TypeError):
                        pass
                if metrics:
                    epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    return epochs_metrics


def aggregate_over_seeds(
    runs_metrics: List[Dict[int, Dict[str, float]]],
    metric_keys: List[str],
) -> Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For each epoch (union across runs), compute mean and std across runs.
    Returns: epochs (sorted), means[metric_key], stds[metric_key].
    """
    all_epochs = set()
    for em in runs_metrics:
        all_epochs.update(em.keys())
    epochs = sorted(all_epochs)
    n_epochs = len(epochs)
    means = {k: np.full(n_epochs, np.nan) for k in metric_keys}
    stds = {k: np.full(n_epochs, np.nan) for k in metric_keys}

    for i, epoch in enumerate(epochs):
        for key in metric_keys:
            values = []
            for em in runs_metrics:
                if epoch in em and em[epoch].get(key) is not None:
                    values.append(em[epoch][key])
            if values:
                means[key][i] = float(np.mean(values))
                stds[key][i] = float(np.std(values)) if len(values) > 1 else 0.0
    return epochs, means, stds


def find_one_experiment_per_model(
    results_dir: Path,
    image_size: int = 128,
    models: Optional[List[str]] = None,
) -> Dict[str, List[Path]]:
    """
    Discover one experiment dir per model (same logic as compare_all_models_all_epochs).
    Returns model_name -> [single_path] so that stability plot works with std=0.
    """
    if models is None:
        models = ["FSQ", "LFQ", "LMB", "SIM_VQ", "VQ"]
    out: Dict[str, List[Path]] = {}

    for model_name in models:
        exp_dir = _find_experiment(results_dir, model_name, image_size)
        if exp_dir is not None:
            out[model_name] = [exp_dir]
    return out


def _find_experiment(results_dir: Path, model_name: str, image_size: int) -> Optional[Path]:
    """One experiment dir for the model (128x128, preferred config)."""
    if model_name == "FSQ":
        candidates_16384 = []
        candidates_other = []
        for exp_dir in (results_dir / "fsq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                levels = config.get("levels", [])
                total = 1
                for L in levels:
                    total *= int(L)
                (candidates_16384 if total == 16384 else candidates_other).append(exp_dir)
            except Exception:
                pass
        for cand in (candidates_16384, candidates_other):
            cand.sort(key=lambda d: _epoch_count(d), reverse=True)
        if candidates_16384:
            return candidates_16384[0]
        if candidates_other:
            return candidates_other[0]
    elif model_name == "LMB":
        candidates_22 = []
        candidates_other = []
        for exp_dir in (results_dir / "lmb").iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "lmb_fixed_init":
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                if "20260122" in exp_dir.name and config.get("batch_size") == 32:
                    candidates_22.append(exp_dir)
                else:
                    candidates_other.append(exp_dir)
            except Exception:
                pass
        for c in (candidates_22, candidates_other):
            c.sort(key=lambda d: _epoch_count(d), reverse=True)
        if candidates_22:
            return candidates_22[0]
        if candidates_other:
            return candidates_other[0]
    elif model_name == "LFQ":
        for exp_dir in (results_dir / "lfq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") == image_size and config.get("codebook_size") == 16384:
                    return exp_dir
            except Exception:
                pass
    elif model_name == "SIM_VQ":
        candidates = []
        for exp_dir in (results_dir / "sim_vq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") == image_size:
                    candidates.append(exp_dir)
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda d: _epoch_count(d), reverse=True)
            return candidates[0]
    elif model_name == "VQ":
        candidates = []
        for exp_dir in (results_dir / "vq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") == image_size:
                    candidates.append(exp_dir)
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda d: _epoch_count(d), reverse=True)
            return candidates[0]
    return None


def _epoch_count(exp_dir: Path) -> int:
    eval_csv = exp_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return 0
    with open(eval_csv) as f:
        return sum(1 for _ in csv.DictReader(f))


# Metrics to plot: (csv_key, ylabel, higher_is_better)
STABILITY_METRICS = [
    ("val_rfid", "rFID (↓)", False),
    ("val_active_codes", "Active codes (↑)", True),
    ("val_perplexity", r"Perplexity / $K_{\mathrm{eff}}$ (↑)", True),
]

MODEL_COLORS = {
    "FSQ": "#1f77b4",
    "LFQ": "#ff7f0e",
    "LMB": "#2ca02c",
    "SIM_VQ": "#9467bd",
    "VQ": "#8c564b",
}


def plot_stability(
    model_aggregates: Dict[str, Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray]]],
    output_path: Path,
) -> None:
    """Plot mean ± std vs epoch for rFID, active codes, perplexity."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for idx, (csv_key, ylabel, _) in enumerate(STABILITY_METRICS):
        ax = axes[idx]
        for model_name, (epochs, means, stds) in model_aggregates.items():
            if csv_key not in means or epochs is None or len(epochs) == 0:
                continue
            m = means[csv_key]
            s = stds[csv_key]
            valid = ~np.isnan(m)
            if not np.any(valid):
                continue
            ep = np.array(epochs)[valid]
            mu = m[valid]
            sigma = s[valid]
            color = MODEL_COLORS.get(model_name, None)
            ax.plot(ep, mu, "o-", label=model_name, color=color, linewidth=2, markersize=4)
            ax.fill_between(
                ep,
                mu - sigma,
                mu + sigma,
                color=color,
                alpha=0.25,
            )
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")
        ax.set_title(ylabel.split("(")[0].strip(), fontsize=12, fontweight="600")
    fig.suptitle(
        "Stability across seeds: mean ± std (rFID, active codes, perplexity)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {output_path}")


def print_fixed_epoch_table(
    model_aggregates: Dict[str, Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray]]],
    fixed_epoch: int,
) -> None:
    """Print mean ± std at a fixed epoch for each metric and model."""
    print(f"\n--- Mean ± std at epoch {fixed_epoch} ---")
    for csv_key, display_name, _ in STABILITY_METRICS:
        print(f"\n{display_name}:")
        for model_name, (epochs, means, stds) in model_aggregates.items():
            if csv_key not in means or fixed_epoch not in epochs:
                continue
            i = epochs.index(fixed_epoch)
            mu = means[csv_key][i]
            sigma = stds[csv_key][i]
            if np.isnan(mu):
                continue
            if csv_key == "val_active_codes":
                print(f"  {model_name}: {mu:.0f} ± {sigma:.0f}")
            else:
                print(f"  {model_name}: {mu:.2f} ± {sigma:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot stability across seeds: mean ± std of rFID, active codes, perplexity."
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file: { 'ModelName': [ 'path/to/exp1', 'path/to/exp2', ... ] }. One path per seed.",
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Used when --config is not set: discover one run per model from results_dir.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("results/plots"),
        help="Output directory for the stability plot.",
    )
    ap.add_argument(
        "--fixed-epoch",
        type=int,
        default=None,
        help="If set, print mean ± std at this epoch for each metric.",
    )
    ap.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size for experiment discovery (when not using --config).",
    )
    args = ap.parse_args()

    # Model -> list of experiment dirs (one per seed)
    if args.config is not None and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
        model_dirs: Dict[str, List[Path]] = {}
        for name, paths in config.items():
            model_dirs[name] = [Path(p).resolve() for p in paths]
    else:
        model_dirs = find_one_experiment_per_model(args.results_dir, args.image_size)

    if not model_dirs:
        print("No experiments found. Use --config with a JSON file or run from repo root with --results-dir results.")
        return

    metric_keys = [m[0] for m in STABILITY_METRICS]
    model_aggregates: Dict[str, Tuple[List[int], Dict[str, np.ndarray], Dict[str, np.ndarray]]] = {}

    for model_name, dirs in model_dirs.items():
        runs_metrics = [load_epoch_metrics(d) for d in dirs]
        runs_metrics = [r for r in runs_metrics if r]
        if not runs_metrics:
            print(f"Warning: no metrics for {model_name}")
            continue
        epochs, means, stds = aggregate_over_seeds(runs_metrics, metric_keys)
        model_aggregates[model_name] = (epochs, means, stds)
        n_seeds = len(runs_metrics)
        print(f"  {model_name}: {n_seeds} run(s), {len(epochs)} epochs")

    if not model_aggregates:
        print("No data to plot.")
        return

    out_path = args.out / "stability_seeds_mean_std.png"
    plot_stability(model_aggregates, out_path)

    if args.fixed_epoch is not None:
        print_fixed_epoch_table(model_aggregates, args.fixed_epoch)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot 128x128 comprehensive metrics comparison (VQ, FSQ, SimVQ, LMB) in a 2x3 grid.
Same style as the original comparison plot, with configurable max epoch (default 12).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


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
                        if value and str(value).strip() and str(value) != "None":
                            metrics[key] = float(value)
                        else:
                            metrics[key] = None
                    except (ValueError, TypeError):
                        metrics[key] = None
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    return epochs_metrics


def find_experiment(model_name: str, image_size: int = 128) -> Optional[Path]:
    """Find experiment directory for a model (128x128)."""
    results_dir = Path("results")
    if model_name == "FSQ":
        candidates_16384 = []
        candidates_other = []
        for exp_dir in (results_dir / "fsq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists():
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
                is_16384 = total == 16384
                n = 0
                if eval_path.exists():
                    with open(eval_path) as ef:
                        n = sum(1 for _ in csv.DictReader(ef))
                (candidates_16384 if is_16384 else candidates_other).append((exp_dir, n))
            except Exception:
                pass
        for cand in (candidates_16384, candidates_other):
            cand.sort(key=lambda x: x[1], reverse=True)
        if candidates_16384:
            return candidates_16384[0][0]
        if candidates_other:
            return candidates_other[0][0]
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
                with open(eval_path) as ef:
                    n = sum(1 for _ in csv.DictReader(ef))
                if "20260122" in exp_dir.name and config.get("batch_size") == 32:
                    candidates_22.append((exp_dir, n))
                else:
                    candidates_other.append((exp_dir, n))
            except Exception:
                pass
        for c in (candidates_22, candidates_other):
            c.sort(key=lambda x: x[1], reverse=True)
        if candidates_22:
            return candidates_22[0][0]
        if candidates_other:
            return candidates_other[0][0]
    elif model_name == "SIM_VQ":
        candidates = []
        for exp_dir in (results_dir / "sim_vq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                n = 0
                if eval_path.exists():
                    with open(eval_path) as ef:
                        n = sum(1 for _ in csv.DictReader(ef))
                candidates.append((exp_dir, n))
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    elif model_name == "VQ":
        candidates = []
        for exp_dir in (results_dir / "vq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                n = 0
                if eval_path.exists():
                    with open(eval_path) as ef:
                        n = sum(1 for _ in csv.DictReader(ef))
                candidates.append((exp_dir, n))
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    return None


def main():
    parser = argparse.ArgumentParser(description="Plot 128x128 comprehensive metrics (VQ, FSQ, SimVQ, LMB).")
    parser.add_argument("--max-epoch", type=int, default=12, help="Plot epochs 1..max_epoch (default: 12)")
    parser.add_argument("-o", "--output", type=Path, default=Path("plots/all_metrics_comparison_128x128.png"))
    args = parser.parse_args()

    # Same 4 models as the original comparison plot (no LFQ, no LMB-Fixed)
    models = ["VQ", "FSQ", "SIM_VQ", "LMB"]
    experiments: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name in models:
        exp_dir = find_experiment(name, 128)
        if exp_dir:
            em = load_epoch_metrics(exp_dir)
            if em:
                experiments[name] = em
                print(f"Loaded {name}: {len(em)} epochs")

    if not experiments:
        print("No experiments found.")
        return

    # 6 metrics in 2x3 grid (matching original comprehensive comparison)
    metrics = [
        ("val_rfid", "rFID (lower is better)", False),
        ("val_psnr", "PSNR (dB) (higher is better)", True),
        ("val_ssim", "SSIM (higher is better)", True),
        ("val_lpips", "LPIPS (lower is better)", False),
        ("val_rec_loss", "Reconstruction Loss (lower is better)", False),
        ("val_codebook_util", "Codebook Utilization (%)", True),
    ]

    # Colors and markers to match original: VQ blue circles, FSQ orange squares, SimVQ green triangles, LMB red diamonds
    colors = {
        "VQ": "#1f77b4",
        "FSQ": "#ff7f0e",
        "SIM_VQ": "#2ca02c",
        "LMB": "#d62728",
    }
    markers = {"VQ": "o", "FSQ": "s", "SIM_VQ": "^", "LMB": "D"}
    # Display name for legend (LMB -> LGQ)
    display_names = {"LMB": "LGQ", "VQ": "VQ", "FSQ": "FSQ", "SIM_VQ": "SIM_VQ"}

    all_epochs = sorted(set().union(*[set(em.keys()) for em in experiments.values()]))
    all_epochs = [e for e in all_epochs if 1 <= e <= args.max_epoch]
    if not all_epochs:
        print(f"No epochs in range [1, {args.max_epoch}].")
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()
    fig.suptitle(
        f"128x128 Image Size - Comprehensive Metrics Comparison (Epochs 1–{args.max_epoch})",
        fontsize=14,
        fontweight="bold",
    )

    for idx, (csv_key, ylabel, _higher_better) in enumerate(metrics):
        ax = axes[idx]
        for model_name, epochs_metrics in experiments.items():
            epochs = []
            values = []
            for e in all_epochs:
                if e not in epochs_metrics:
                    continue
                v = epochs_metrics[e].get(csv_key)
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    epochs.append(e)
                    values.append(float(v))
            if epochs and values:
                ax.plot(
                    epochs,
                    values,
                    marker=markers.get(model_name, "o"),
                    linestyle="-",
                    label=display_names.get(model_name, model_name),
                    color=colors.get(model_name),
                    linewidth=2,
                    markersize=6,
                    alpha=0.9,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

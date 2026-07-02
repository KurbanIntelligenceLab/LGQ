#!/usr/bin/env python3
"""
Plot comparison between FSQ 16K and LMB perchannel_fair (both with same codebook structure).
"""

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
                        if value and str(value).strip() and str(value) != "None" and str(value) != "nan":
                            metrics[key] = float(value)
                        else:
                            metrics[key] = None
                    except (ValueError, TypeError):
                        metrics[key] = None
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    return epochs_metrics


def find_fsq_16k() -> Optional[Path]:
    """Find FSQ 16K experiment."""
    results_dir = Path("results")
    for exp_dir in (results_dir / "fsq").iterdir():
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("image_size") != 128:
                continue
            levels = config.get("levels", [])
            total = 1
            for L in levels:
                total *= int(L)
            if total == 16384:
                return exp_dir
        except Exception:
            pass
    return None


def main():
    # Find experiments
    fsq_dir = find_fsq_16k()
    lmb_fair_dir = Path("results/lmb/lmb_ablation_perchannel_fair")
    
    experiments = {}
    
    if fsq_dir and fsq_dir.exists():
        em = load_epoch_metrics(fsq_dir)
        if em:
            experiments["FSQ 16K"] = em
            print(f"Loaded FSQ 16K: {len(em)} epochs from {fsq_dir}")
    
    if lmb_fair_dir.exists():
        em = load_epoch_metrics(lmb_fair_dir)
        if em:
            experiments["LMB Per-Channel Fair"] = em
            print(f"Loaded LMB Per-Channel Fair: {len(em)} epochs")
    
    if len(experiments) < 2:
        print("Need both experiments to compare.")
        return
    
    # Metrics to plot
    metrics = [
        ("val_rfid", "rFID (↓)", False),
        ("val_psnr", "PSNR (dB) (↑)", True),
        ("val_ssim", "SSIM (↑)", True),
        ("val_lpips", "LPIPS (↓)", False),
        ("val_rec_loss", "Rec Loss (↓)", False),
        ("val_active_codes", "Active Codes (↑)", True),
        ("val_codebook_util", "Codebook Util (%) (↑)", True),
        ("val_perplexity", "Perplexity (↑)", True),
    ]
    
    colors = {
        "FSQ 16K": "#1f77b4",
        "LMB Per-Channel Fair": "#2ca02c",
    }
    
    all_epochs = sorted(set().union(*[set(em.keys()) for em in experiments.values()]))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    fig.suptitle("FSQ vs LMB Per-Channel Fair Comparison (16K codebook, 128×128)", fontsize=14, fontweight="bold")
    
    for idx, (csv_key, ylabel, higher_better) in enumerate(metrics):
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
                    "o-",
                    label=model_name,
                    color=colors.get(model_name, None),
                    linewidth=2,
                    markersize=8,
                    alpha=0.9,
                )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.2, linestyle="--")
    
    plt.tight_layout()
    out = Path("results/plots/fsq_vs_lmb_perchannel_fair.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved {out}")
    
    # Also print summary table
    print("\n" + "="*80)
    print("Summary at Epoch 1 (Corrected Metrics)")
    print("="*80)
    print(f"{'Metric':<25} {'FSQ 16K':<20} {'LMB Per-Channel Fair':<20}")
    print("-"*65)
    
    for csv_key, ylabel, _ in metrics:
        fsq_val = experiments.get("FSQ 16K", {}).get(1, {}).get(csv_key)
        lmb_val = experiments.get("LMB Per-Channel Fair", {}).get(1, {}).get(csv_key)
        
        fsq_str = f"{fsq_val:.2f}" if fsq_val is not None else "N/A"
        lmb_str = f"{lmb_val:.2f}" if lmb_val is not None else "N/A"
        
        metric_name = ylabel.split(" (")[0]
        print(f"{metric_name:<25} {fsq_str:<20} {lmb_str:<20}")


if __name__ == "__main__":
    main()

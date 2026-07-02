"""
Plot utilization / perplexity / rec_loss vs training step for different codebook sizes.
Addresses reviewer concern: show that metrics converge and comparisons are fair.

Usage:
    python scripts/plot_convergence.py --results-dir results/lmb --output convergence.pdf
"""

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]


def load_run(run_dir):
    config_path = os.path.join(run_dir, "config.json")
    csv_path = os.path.join(run_dir, "train_metrics.csv")
    if not os.path.exists(config_path) or not os.path.exists(csv_path):
        return None

    with open(config_path) as f:
        config = json.load(f)
    codebook_size = config.get("num_bins") or config.get("codebook_size")
    if codebook_size is None:
        return None

    steps, active, perplexity, rec_loss = [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row["global_step"]))
            active.append(int(row["active_codes"]))
            perplexity.append(float(row["perplexity"]))
            rec_loss.append(float(row["rec_loss"]))

    if not steps:
        return None

    return {
        "codebook_size": codebook_size,
        "run_name": os.path.basename(run_dir),
        "steps": np.array(steps),
        "active_codes": np.array(active),
        "utilization": 100 * np.array(active) / codebook_size,
        "perplexity": np.array(perplexity),
        "rec_loss": np.array(rec_loss),
    }


def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/lmb")
    parser.add_argument("--pattern", default="lmb_cb")
    parser.add_argument("--output", default="convergence.pdf")
    parser.add_argument("--smooth-window", type=int, default=50)
    args = parser.parse_args()

    runs = []
    for name in sorted(os.listdir(args.results_dir)):
        if not name.startswith(args.pattern):
            continue
        run_dir = os.path.join(args.results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        info = load_run(run_dir)
        if info and max(info["active_codes"]) > 1:
            runs.append(info)
            print(f"  {info['run_name']}: cb={info['codebook_size']}, "
                  f"{len(info['steps'])} logged steps")

    if not runs:
        print("No valid (non-collapsed) runs found.")
        sys.exit(1)

    runs.sort(key=lambda r: r["codebook_size"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    metrics = [
        ("utilization", "Utilization (%)", axes[0]),
        ("perplexity", "Perplexity", axes[1]),
        ("rec_loss", "Reconstruction Loss", axes[2]),
    ]

    for i, run in enumerate(runs):
        color = COLORS[i % len(COLORS)]
        label = f"CB={run['codebook_size']:,}"
        w = args.smooth_window

        for key, ylabel, ax in metrics:
            y = run[key]
            x = run["steps"]
            if len(y) > w:
                y_smooth = smooth(y, w)
                x_smooth = x[w - 1:][:len(y_smooth)]
            else:
                y_smooth = y
                x_smooth = x

            ax.plot(x_smooth, y_smooth, color=color, label=label, linewidth=1.5, alpha=0.85)

    for key, ylabel, ax in metrics:
        ax.set_xlabel("Training Step", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Codebook Utilization", fontsize=12)
    axes[1].set_title("Perplexity", fontsize=12)
    axes[2].set_title("Reconstruction Loss", fontsize=12)

    plt.suptitle("LGQ Convergence Across Codebook Sizes", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

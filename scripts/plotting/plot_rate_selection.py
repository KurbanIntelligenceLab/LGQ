"""
Plot LGQ automatic rate selection: utilization % vs codebook size.
Shows that LGQ adapts utilization — high % with small codebooks, lower with large ones.

Usage:
    python scripts/plot_rate_selection.py [--results-dir results/lmb] [--output rate_selection.pdf]
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


def load_run_utilization(run_dir):
    """Load codebook size and active_codes trajectory from a run directory."""
    config_path = os.path.join(run_dir, "config.json")
    csv_path = os.path.join(run_dir, "train_metrics.csv")

    if not os.path.exists(config_path) or not os.path.exists(csv_path):
        return None

    with open(config_path) as f:
        config = json.load(f)

    codebook_size = config.get("num_bins") or config.get("codebook_size")
    if codebook_size is None:
        return None

    # Read active_codes from CSV
    epochs = []
    active_codes = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row["epoch"])
            ac = int(row["active_codes"])
            epochs.append(ep)
            active_codes.append(ac)

    if not active_codes:
        return None

    return {
        "codebook_size": codebook_size,
        "run_name": os.path.basename(run_dir),
        "epochs": epochs,
        "active_codes": active_codes,
        "final_active": active_codes[-1],
        "max_active": max(active_codes),
        "final_epoch": epochs[-1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/lmb",
                        help="Directory containing LGQ run subdirectories")
    parser.add_argument("--pattern", default="lmb_cb",
                        help="Prefix filter for run directories")
    parser.add_argument("--output", default="rate_selection.pdf")
    parser.add_argument("--use-max", action="store_true",
                        help="Use max active codes instead of final")
    args = parser.parse_args()

    # Collect all runs
    runs = []
    for name in sorted(os.listdir(args.results_dir)):
        if not name.startswith(args.pattern):
            continue
        run_dir = os.path.join(args.results_dir, name)
        if not os.path.isdir(run_dir):
            continue
        info = load_run_utilization(run_dir)
        if info and info["final_active"] > 1:  # skip collapsed runs
            runs.append(info)
            print(f"  {info['run_name']}: cb={info['codebook_size']}, "
                  f"active={info['final_active']}/{info['codebook_size']} "
                  f"({100*info['final_active']/info['codebook_size']:.1f}%), "
                  f"epoch={info['final_epoch']}")

    if not runs:
        print("No valid (non-collapsed) runs found. Exiting.")
        sys.exit(1)

    # Sort by codebook size
    runs.sort(key=lambda r: r["codebook_size"])

    cb_sizes = [r["codebook_size"] for r in runs]
    if args.use_max:
        utilization = [100 * r["max_active"] / r["codebook_size"] for r in runs]
        active = [r["max_active"] for r in runs]
    else:
        utilization = [100 * r["final_active"] / r["codebook_size"] for r in runs]
        active = [r["final_active"] for r in runs]

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: Utilization %
    ax1.plot(cb_sizes, utilization, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Codebook Size", fontsize=12)
    ax1.set_ylabel("Utilization (%)", fontsize=12)
    ax1.set_title("LGQ Automatic Rate Selection", fontsize=13)
    ax1.set_ylim(0, 105)
    ax1.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Full utilization")
    ax1.set_xticks(cb_sizes)
    ax1.set_xticklabels([str(s) for s in cb_sizes], fontsize=10)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Annotate points
    for x, y, a in zip(cb_sizes, utilization, active):
        ax1.annotate(f"{a}", (x, y), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=9, color="#1565C0")

    # Right: Active codes (absolute)
    ax2.plot(cb_sizes, active, "s-", color="#4CAF50", linewidth=2, markersize=8)
    ax2.plot(cb_sizes, cb_sizes, "--", color="gray", alpha=0.5, label="y = x (full)")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log", base=2)
    ax2.set_xlabel("Codebook Size", fontsize=12)
    ax2.set_ylabel("Active Codes", fontsize=12)
    ax2.set_title("Active Codes vs Codebook Size", fontsize=13)
    ax2.set_xticks(cb_sizes)
    ax2.set_xticklabels([str(s) for s in cb_sizes], fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

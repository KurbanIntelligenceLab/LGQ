#!/usr/bin/env python3
"""
Plot training metrics for all LMB ablation experiments.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def forward_fill(values):
    filled = []
    last = None
    for val in values:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            filled.append(last if last is not None else np.nan)
        else:
            filled.append(val)
            last = val
    return filled


def smooth_curve(values, weight=0.95):
    if not values:
        return values
    smoothed = [values[0]]
    for val in values[1:]:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            smoothed.append(smoothed[-1])
        else:
            smoothed.append(weight * smoothed[-1] + (1 - weight) * val)
    return smoothed


def downsample(steps, values, max_points=200):
    if len(steps) <= max_points:
        return steps, values
    indices = np.linspace(0, len(steps) - 1, max_points, dtype=int)
    return [steps[i] for i in indices], [values[i] for i in indices]


def load_train_metrics(csv_path, max_step=None):
    data = {
        "steps": [],
        "rec_loss": [],
        "total_loss": [],
        "vq_loss": [],
        "perplexity": [],
        "active_codes": [],
    }

    if not csv_path.exists():
        return data

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row.get("global_step", 0))
            except (ValueError, TypeError):
                continue

            if max_step is not None and step > max_step:
                continue

            data["steps"].append(step)

            def safe_float(key):
                val = row.get(key, "")
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return np.nan

            rec = safe_float("rec_loss")
            if rec > 10 or rec < 0:
                rec = np.nan

            data["rec_loss"].append(rec)
            data["total_loss"].append(safe_float("total_loss"))
            data["vq_loss"].append(safe_float("vq_loss"))
            data["perplexity"].append(safe_float("perplexity"))

            try:
                data["active_codes"].append(int(row.get("active_codes", 0)))
            except (ValueError, TypeError):
                data["active_codes"].append(np.nan)

    return data


def find_ablation_runs(base_dir, include_keywords=None, exclude_keywords=None):
    runs = sorted([p for p in base_dir.glob("lmb_ablation_*") if p.is_dir()])
    if include_keywords:
        runs = [
            r for r in runs
            if any(keyword in r.name for keyword in include_keywords)
        ]
    if exclude_keywords:
        runs = [
            r for r in runs
            if not any(keyword in r.name for keyword in exclude_keywords)
        ]
    return runs


def plot_ablations(base_dir, output_path, max_step=None, max_points=200,
                   include_keywords=None, exclude_keywords=None):
    runs = find_ablation_runs(base_dir, include_keywords, exclude_keywords)
    if not runs:
        print(f"No ablation runs found in {base_dir}")
        return

    run_data = {}
    for run in runs:
        csv_path = run / "train_metrics.csv"
        data = load_train_metrics(csv_path, max_step=max_step)
        if data["steps"]:
            run_data[run.name] = data

    if not run_data:
        print("No train_metrics.csv found for ablations.")
        return

    colors = plt.cm.tab20(np.linspace(0, 1, len(run_data)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("LMB Ablation Experiments - Training Metrics", fontsize=14, fontweight="bold")

    metric_specs = [
        ("rec_loss", "Reconstruction Loss (lower is better)", axes[0, 0]),
        ("vq_loss", "VQ Loss", axes[0, 1]),
        ("active_codes", "Active Codes", axes[1, 0]),
        ("perplexity", "Perplexity", axes[1, 1]),
    ]

    handles = []
    labels = []

    for (metric, ylabel, ax) in metric_specs:
        for (idx, (run_name, data)) in enumerate(run_data.items()):
            steps = data["steps"]
            values = forward_fill(data[metric])

            if not steps or not values:
                continue

            steps_ds, values_ds = downsample(steps, values, max_points=max_points)
            values_smooth = smooth_curve(values_ds, weight=0.95)

            line, = ax.plot(
                steps_ds,
                values_smooth,
                linewidth=1.6,
                alpha=0.85,
                color=colors[idx],
            )
            if metric == "rec_loss":
                handles.append(line)
                labels.append(run_name)

        ax.set_xlabel("Training Steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.2, linestyle="--")

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        frameon=True,
    )

    fig.tight_layout(rect=[0, 0, 0.82, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot LMB ablation training metrics.")
    parser.add_argument("--results-dir", type=str, default="results/lmb",
                        help="Directory containing LMB ablation runs")
    parser.add_argument("--output", type=str, default="plots/lmb_ablation_train_metrics.png",
                        help="Output plot path")
    parser.add_argument("--max-step", type=int, default=None,
                        help="Max training step to include")
    parser.add_argument("--max-points", type=int, default=200,
                        help="Downsample each run to this many points")
    parser.add_argument("--include", nargs="*", default=None,
                        help="Only include runs containing any of these keywords")
    parser.add_argument("--exclude", nargs="*", default=None,
                        help="Exclude runs containing any of these keywords")
    args = parser.parse_args()

    plot_ablations(
        Path(args.results_dir),
        Path(args.output),
        args.max_step,
        args.max_points,
        include_keywords=args.include,
        exclude_keywords=args.exclude,
    )


if __name__ == "__main__":
    main()

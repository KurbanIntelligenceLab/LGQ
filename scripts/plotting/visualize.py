#!/usr/bin/env python3
"""
Visualization tool for comparing training metrics across quantization models and runs.

Reads train_metrics.csv files from results/<model>/<run>/ directories and plots
training metrics with different lines for each run (wandb style).

Supports:
- Comparing different models (FSQ vs VQ vs LMB etc.)
- Comparing different runs of the same model
- Custom run selection via glob patterns

Usage:
    # Compare all models (latest run of each)
    python scripts/visualize.py --results-dir results/
    
    # Compare all runs of a specific model
    python scripts/visualize.py --results-dir results/fsq/
    
    # Compare specific runs
    python scripts/visualize.py --runs results/fsq/run1 results/vq/run2
    
    # Customize plots
    python scripts/visualize.py --metrics rec_loss perplexity --smoothing 0.9
"""

import argparse
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package: {e.name}")
    print("Install with: pip install matplotlib numpy")
    exit(1)


def load_metrics_from_csv(csv_path: Path) -> dict:
    """Load metrics from a CSV file."""
    metrics = defaultdict(list)
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value and value != 'None':
                    try:
                        metrics[key].append(float(value))
                    except ValueError:
                        pass
    
    return dict(metrics)


def load_config(run_dir: Path) -> Optional[dict]:
    """Load config.json from run directory."""
    config_path = run_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return None


def get_run_label(run_dir: Path, config: Optional[dict] = None) -> str:
    """Generate a label for a run based on its directory and config."""
    if config and "run_name" in config:
        return config["run_name"]
    
    # Use directory structure: model/run_name or just run_name
    parts = run_dir.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return parts[-1]


def find_all_runs(results_dir: Path) -> List[Path]:
    """
    Find all run directories under results_dir.
    
    Handles both:
    - results/<model>/train_metrics.csv (old format)
    - results/<model>/<run>/train_metrics.csv (new format)
    """
    runs = []
    
    for path in results_dir.rglob("train_metrics.csv"):
        runs.append(path.parent)
    
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_runs_per_model(results_dir: Path) -> Dict[str, Path]:
    """Find the latest run for each model type."""
    all_runs = find_all_runs(results_dir)
    
    model_runs = {}
    for run_dir in all_runs:
        config = load_config(run_dir)
        if config:
            model = config.get("model", run_dir.parent.name)
        else:
            # Try to infer from directory name
            model = run_dir.parent.name if run_dir.parent != results_dir else run_dir.name
        
        if model not in model_runs:
            model_runs[model] = run_dir
    
    return model_runs


def get_all_model_results(
    results_dir: Path,
    specific_runs: Optional[List[str]] = None,
    compare_mode: str = "latest",  # "latest", "all", or "specific"
) -> Dict[str, dict]:
    """
    Get metrics from run directories.
    
    Args:
        results_dir: Root results directory
        specific_runs: List of specific run paths to compare
        compare_mode: "latest" for latest run per model, "all" for all runs
    """
    results = {}
    
    if specific_runs:
        # Use specific runs provided
        for run_path in specific_runs:
            run_dir = Path(run_path)
            csv_path = run_dir / "train_metrics.csv"
            if csv_path.exists():
                metrics = load_metrics_from_csv(csv_path)
                if metrics:
                    config = load_config(run_dir)
                    label = get_run_label(run_dir, config)
                    results[label] = metrics
    elif compare_mode == "all":
        # Get all runs
        all_runs = find_all_runs(results_dir)
        for run_dir in all_runs:
            csv_path = run_dir / "train_metrics.csv"
            metrics = load_metrics_from_csv(csv_path)
            if metrics:
                config = load_config(run_dir)
                label = get_run_label(run_dir, config)
                results[label] = metrics
    else:
        # Get latest run per model (default)
        model_runs = find_latest_runs_per_model(results_dir)
        for model, run_dir in model_runs.items():
            csv_path = run_dir / "train_metrics.csv"
            metrics = load_metrics_from_csv(csv_path)
            if metrics:
                config = load_config(run_dir)
                label = model if compare_mode == "latest" else get_run_label(run_dir, config)
                results[label] = metrics
    
    return results


def get_common_metrics(results: dict) -> list:
    """Get metrics that exist in all models."""
    if not results:
        return []
    
    common = None
    for model_metrics in results.values():
        metric_set = set(model_metrics.keys())
        if common is None:
            common = metric_set
        else:
            common = common.intersection(metric_set)
    
    skip_metrics = {'epoch', 'global_step', 'batch_idx'}
    return sorted([m for m in common if m not in skip_metrics])


def smooth_curve(y: list, weight: float = 0.9) -> list:
    """Apply exponential moving average smoothing."""
    if weight <= 0 or weight >= 1:
        return y
    
    smoothed = []
    last = y[0]
    for point in y:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_metric(
    results: dict,
    metric: str,
    x_axis: str = "global_step",
    smoothing: float = 0.9,
    save_path: Path = None,
):
    """Plot a single metric for all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use colormap for distinct colors
    model_names = sorted(results.keys())
    n_models = len(model_names)
    
    if n_models <= 10:
        cmap = plt.colormaps['tab10']
    elif n_models <= 12:
        cmap = plt.colormaps['Set3']
    else:
        cmap = plt.colormaps['tab20']
    
    colors = [mcolors.to_hex(cmap(i / max(n_models - 1, 1))) for i in range(n_models)]
    
    for idx, model_name in enumerate(model_names):
        model_metrics = results[model_name]
        if metric not in model_metrics:
            continue
        
        y = model_metrics[metric]
        
        if x_axis in model_metrics:
            x = model_metrics[x_axis]
        else:
            x = list(range(len(y)))
        
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        
        y_smooth = smooth_curve(y, smoothing)
        
        color = colors[idx]
        ax.plot(x, y_smooth, label=model_name, color=color, linewidth=1.5, alpha=0.9)
    
    ax.set_xlabel(x_axis.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(f'{metric.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path.absolute()}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training metrics across models and runs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory containing model result subdirectories")
    parser.add_argument("--runs", type=str, nargs="+", default=None,
                        help="Specific run directories to compare")
    parser.add_argument("--compare-all", action="store_true",
                        help="Compare all runs (not just latest per model)")
    parser.add_argument("--metrics", type=str, nargs="+", default=None,
                        help="Specific metrics to plot (default: all common metrics)")
    parser.add_argument("--x-axis", type=str, default="global_step",
                        choices=["global_step", "epoch"],
                        help="X-axis for plots")
    parser.add_argument("--smoothing", type=float, default=0.9,
                        help="Smoothing weight (0-1, 0=no smoothing)")
    parser.add_argument("--save", action="store_true", default=True,
                        help="Save plots to files")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save plots")
    parser.add_argument("--show", action="store_true", default=True,
                        help="Show plots interactively")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't show plots interactively")
    parser.add_argument("--output-dir", type=str, default="plots",
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    save = args.save and not args.no_save
    show = args.show and not args.no_show
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Loading results from: {results_dir}")
    
    # Determine comparison mode
    compare_mode = "all" if args.compare_all else "latest"
    
    results = get_all_model_results(
        results_dir,
        specific_runs=args.runs,
        compare_mode=compare_mode,
    )
    
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} runs:")
    for name in sorted(results.keys()):
        print(f"  - {name}")
    
    # Determine metrics to plot
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = get_common_metrics(results)
    
    if not metrics:
        print("No common metrics found across runs!")
        return
    
    print(f"\nPlotting metrics: {', '.join(metrics)}")
    
    output_dir = Path(args.output_dir)
    saved_files = []
    
    for metric in metrics:
        save_path = output_dir / f"{metric}.png" if save else None
        plot_metric(
            results,
            metric,
            x_axis=args.x_axis,
            smoothing=args.smoothing,
            save_path=save_path,
        )
        if save_path:
            saved_files.append(save_path)
    
    if save and saved_files:
        print(f"\nSaved {len(saved_files)} plots to {output_dir.absolute()}/")
    
    if show:
        plt.show()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot evaluation metrics for LMB ablation experiments, grouped by category.
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_eval_metrics(csv_path):
    """Load evaluation metrics from CSV file."""
    metrics = defaultdict(list)
    
    if not csv_path.exists():
        return None
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if value and value != "None" and value != "":
                    try:
                        metrics[key].append(float(value))
                    except (ValueError, TypeError):
                        pass
    
    # Get the last (most recent) values for each metric
    result = {}
    for key, values in metrics.items():
        if values:
            result[key] = values[-1]  # Last epoch value
    
    return result if result else None


def get_ablation_categories():
    """Define ablation experiment categories."""
    return {
        "Regularization": {
            "lmb_ablation_reg_none": "None",
            "lmb_ablation_reg_weak": "Weak",
            "lmb_ablation_reg_strong": "Strong",
            "lmb_ablation_reg_peak_only": "Peak Only",
            "lmb_ablation_reg_bins_only": "Bins Only",
        },
        "Temperature Schedule": {
            "lmb_ablation_tau_fast": "Fast (1.0→0.05)",
            "lmb_ablation_tau_slow": "Slow (1.0→0.2)",
            "lmb_ablation_tau_fixed": "Fixed (1.0)",
        },
        "Codebook Size": {
            "lmb_ablation_cb4k": "4K",
            "lmb_ablation_cb8k": "8K",
            "lmb_ablation_cb16k": "16K",
            "lmb_ablation_cb32k": "32K",
            "lmb_ablation_cb65k": "65K",
        },
        "Flattening": {
            "lmb_ablation_flattened": "Flattened",
            "lmb_ablation_perchannel_fair": "Per-Channel (16,16,8,8)",
        },
    }


def plot_category_ablations(base_dir, category_name, experiments, output_path):
    """Plot evaluation metrics for a specific category of ablations."""
    results = {}
    
    for exp_name, display_name in experiments.items():
        exp_dir = base_dir / exp_name
        if not exp_dir.exists():
            continue
        
        csv_path = exp_dir / "eval_metrics.csv"
        metrics = load_eval_metrics(csv_path)
        if metrics:
            results[display_name] = metrics
    
    if not results:
        print(f"No data found for category: {category_name}")
        return
    
    # Define metrics to plot
    metric_configs = [
        ("val_rfid", "rFID (↓)", False),
        ("val_psnr", "PSNR (dB) (↑)", True),
        ("val_ssim", "SSIM (↑)", True),
        ("val_lpips", "LPIPS (↓)", False),
        ("val_rec_loss", "Rec Loss (↓)", False),
        ("val_codebook_util", "Codebook Util (%) (↑)", True),
    ]
    
    n_metrics = len(metric_configs)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f"LMB Ablations: {category_name}", fontsize=14, fontweight="bold")
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    exp_names = sorted(results.keys())
    
    for idx, (metric_key, ylabel, higher_better) in enumerate(metric_configs):
        ax = axes[idx]
        
        values = []
        labels = []
        exp_colors = []
        
        for exp_idx, exp_name in enumerate(exp_names):
            if metric_key in results[exp_name]:
                val = results[exp_name][metric_key]
                # Skip invalid values
                if not (np.isnan(val) or np.isinf(val) or val < 0 or val > 1000):
                    values.append(val)
                    labels.append(exp_name)
                    exp_colors.append(colors[exp_idx])
        
        if values:
            bars = ax.bar(range(len(values)), values, color=exp_colors, alpha=0.7, edgecolor='black', linewidth=1)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}' if val < 1 else f'{val:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        ax.set_title(ylabel, fontsize=11)
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_all_ablations_comparison(base_dir, output_path):
    """Plot a comparison of all ablation experiments."""
    categories = get_ablation_categories()
    all_experiments = {}
    for cat_exps in categories.values():
        all_experiments.update(cat_exps)
    
    results = {}
    for exp_name, display_name in all_experiments.items():
        exp_dir = base_dir / exp_name
        if not exp_dir.exists():
            continue
        
        csv_path = exp_dir / "eval_metrics.csv"
        metrics = load_eval_metrics(csv_path)
        if metrics:
            results[display_name] = metrics
    
    if not results:
        print("No ablation data found")
        return
    
    # Key metrics for comparison
    metric_configs = [
        ("val_rfid", "rFID (↓)", False),
        ("val_psnr", "PSNR (dB) (↑)", True),
        ("val_ssim", "SSIM (↑)", True),
        ("val_codebook_util", "Codebook Util (%) (↑)", True),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    fig.suptitle("LMB Ablation Experiments - Evaluation Metrics Comparison", fontsize=14, fontweight="bold")
    
    n_exps = len(results)
    colors = plt.cm.tab20(np.linspace(0, 1, n_exps))
    exp_names = sorted(results.keys())
    
    for idx, (metric_key, ylabel, higher_better) in enumerate(metric_configs):
        ax = axes[idx]
        
        values = []
        labels = []
        exp_colors = []
        
        for exp_idx, exp_name in enumerate(exp_names):
            if metric_key in results[exp_name]:
                val = results[exp_name][metric_key]
                if not (np.isnan(val) or np.isinf(val) or val < 0 or val > 1000):
                    values.append(val)
                    labels.append(exp_name)
                    exp_colors.append(colors[exp_idx])
        
        if values:
            bars = ax.barh(range(len(values)), values, color=exp_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel(ylabel, fontsize=10)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{val:.3f}' if val < 1 else f'{val:.1f}',
                       ha='left', va='center', fontsize=7)
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot LMB ablation evaluation metrics.")
    parser.add_argument("--results-dir", type=str, default="results/lmb",
                        help="Directory containing LMB ablation runs")
    parser.add_argument("--output-dir", type=str, default="results/plots",
                        help="Output directory for plots")
    parser.add_argument("--category", type=str, default=None,
                        choices=["Regularization", "Temperature Schedule", "Codebook Size", "Flattening"],
                        help="Plot specific category only")
    parser.add_argument("--all", action="store_true",
                        help="Plot all ablations comparison")
    args = parser.parse_args()
    
    base_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    
    categories = get_ablation_categories()
    
    if args.all:
        plot_all_ablations_comparison(base_dir, output_dir / "lmb_ablation_all_comparison.png")
    elif args.category:
        if args.category in categories:
            output_path = output_dir / f"lmb_ablation_{args.category.lower().replace(' ', '_')}.png"
            plot_category_ablations(base_dir, args.category, categories[args.category], output_path)
        else:
            print(f"Unknown category: {args.category}")
    else:
        # Plot all categories
        for category_name, experiments in categories.items():
            output_path = output_dir / f"lmb_ablation_{category_name.lower().replace(' ', '_')}.png"
            plot_category_ablations(base_dir, category_name, experiments, output_path)
        
        # Also create overall comparison
        plot_all_ablations_comparison(base_dir, output_dir / "lmb_ablation_all_comparison.png")


if __name__ == "__main__":
    main()

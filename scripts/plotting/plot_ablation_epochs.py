#!/usr/bin/env python3
"""
Plot LMB ablation studies over epochs - one plot per ablation category.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

plt.style.use('seaborn-v0_8-whitegrid')

def load_all_epochs_metrics(experiment_dir: Path) -> Optional[Dict[int, Dict[str, float]]]:
    """Load metrics from ALL epochs in eval_metrics.csv."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return None
    
    epochs_metrics = {}
    
    with open(eval_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('epoch'):
                continue
            try:
                epoch = int(row['epoch'])
                metrics = {}
                for key, value in row.items():
                    if key not in ['epoch', 'global_step']:
                        try:
                            if value and value.strip() and value != 'None':
                                metrics[key] = float(value)
                            else:
                                metrics[key] = None
                        except (ValueError, TypeError):
                            metrics[key] = None
                metrics['epoch'] = epoch
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    
    return epochs_metrics if epochs_metrics else None


def plot_ablation_over_epochs(
    category_name: str,
    experiments: Dict[str, Dict[int, Dict[str, float]]],
    output_path: Path,
    metrics_to_plot: List[Tuple[str, str, bool]] = None
):
    """Plot metrics over epochs for a single ablation category."""
    
    if metrics_to_plot is None:
        metrics_to_plot = [
            ("val_rfid", "rFID (↓)", False),
            ("val_psnr", "PSNR (dB) (↑)", True),
            ("val_ssim", "SSIM (↑)", True),
            ("val_lpips", "LPIPS (↓)", False),
            ("val_rec_loss", "Rec Loss (↓)", False),
            ("val_codebook_util", "Codebook Util (%) (↑)", True),
        ]
    
    n_metrics = len(metrics_to_plot)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    
    for idx, (metric_key, metric_name, higher_is_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for (exp_name, epochs_data), color in zip(experiments.items(), colors):
            epochs = sorted(epochs_data.keys())
            values = []
            valid_epochs = []
            
            for epoch in epochs:
                val = epochs_data[epoch].get(metric_key)
                if val is not None:
                    values.append(val)
                    valid_epochs.append(epoch)
            
            if valid_epochs:
                ax.plot(valid_epochs, values, 'o-', label=exp_name, color=color, 
                       markersize=6, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=8, loc='best')
        
        # Set x-axis to integer epochs
        if experiments:
            all_epochs = set()
            for epochs_data in experiments.values():
                all_epochs.update(epochs_data.keys())
            if all_epochs:
                ax.set_xlim(min(all_epochs) - 0.5, max(all_epochs) + 0.5)
                ax.set_xticks(sorted(all_epochs))
    
    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'LMB Ablation: {category_name} - Metrics Over Epochs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    results_dir = Path("results/lmb")
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)
    
    # Define ablation categories
    categories = {
        "Flattening": {
            "lmb_ablation_flattened": "Flattened",
            "lmb_ablation_perchannel": "Per-Channel",
        },
        "Temperature": {
            "lmb_ablation_tau_fast": "Fast (1.0→0.05)",
            "lmb_ablation_tau_slow": "Slow (1.0→0.2)",
            "lmb_ablation_tau_fixed": "Fixed (1.0)",
        },
        "Regularization": {
            "lmb_ablation_reg_none": "None",
            "lmb_ablation_reg_weak": "Weak",
            "lmb_ablation_reg_strong": "Strong",
            "lmb_ablation_reg_peak_only": "Peak Only",
            "lmb_ablation_reg_bins_only": "Bins Only",
        },
        "Codebook_Size": {
            "lmb_ablation_cb4k": "4K",
            "lmb_ablation_cb8k": "8K",
            "lmb_ablation_cb16k": "16K",
            "lmb_ablation_cb32k": "32K",
            "lmb_ablation_cb65k": "65K",
        },
        "Initialization": {
            "lmb_ablation_init_uniform": "Uniform",
            "lmb_ablation_init_gaussian": "Gaussian",
        },
    }
    
    # Also add regularization grid
    reg_grid_experiments = {}
    for exp_dir in results_dir.iterdir():
        if exp_dir.is_dir() and "reg_grid" in exp_dir.name:
            name = exp_dir.name
            if "peak002" in name:
                peak = "0.002"
            elif "peak005" in name:
                peak = "0.005"
            elif "peak010" in name:
                peak = "0.010"
            else:
                peak = "?"
            
            if "bins002" in name:
                bins = "0.002"
            elif "bins005" in name:
                bins = "0.005"
            elif "bins010" in name:
                bins = "0.010"
            else:
                bins = "?"
            
            reg_grid_experiments[exp_dir.name] = f"({peak},{bins})"
    
    if reg_grid_experiments:
        categories["Regularization_Grid"] = reg_grid_experiments
    
    # Generate plots for each category
    for category_name, experiment_map in categories.items():
        experiments_data = {}
        
        for exp_dir_name, display_name in experiment_map.items():
            exp_dir = results_dir / exp_dir_name
            if not exp_dir.exists():
                continue
            
            epochs_dict = load_all_epochs_metrics(exp_dir)
            if epochs_dict:
                experiments_data[display_name] = epochs_dict
        
        if experiments_data:
            output_path = output_dir / f"lmb_ablation_{category_name.lower()}_epochs.png"
            plot_ablation_over_epochs(category_name.replace("_", " "), experiments_data, output_path)
    
    print(f"\nGenerated {len(categories)} ablation epoch plots")


if __name__ == "__main__":
    main()

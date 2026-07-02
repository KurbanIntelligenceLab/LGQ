#!/usr/bin/env python3
"""
Plot codebook size scaling - one metric per plot, codebook size on x-axis, lines for each model.
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional

plt.style.use('seaborn-v0_8-whitegrid')

def load_metrics_at_epoch(experiment_dir: Path, target_epoch: int = 1) -> Optional[Dict[str, float]]:
    """Load metrics from a specific epoch in eval_metrics.csv."""
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
    
    if not epochs_metrics:
        return None
    
    # Return requested epoch or latest if not available
    if target_epoch in epochs_metrics:
        return epochs_metrics[target_epoch]
    return None


def main():
    results_dir = Path("results")
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)
    
    # Define models and their codebook size experiments
    models_config = {
        "LMB": ("lmb", "lmb_ablation_cb"),
        "FSQ": ("fsq", "fsq_cb"),
        "LFQ": ("lfq", "lfq_cb"),
        "VQ": ("vq", "vq_cb"),
        "SIM_VQ": ("sim_vq", "sim_vq_cb"),
    }
    
    codebook_sizes = {
        "4K": (4096, "4k"),
        "8K": (8192, "8k"), 
        "16K": (16384, "16k"),
        "32K": (32768, "32k"),
        "65K": (65536, "65k"),
    }
    
    # Metrics to plot
    metrics_to_plot = [
        ("val_rfid", "rFID", "lower is better"),
        ("val_psnr", "PSNR (dB)", "higher is better"),
        ("val_ssim", "SSIM", "higher is better"),
        ("val_lpips", "LPIPS", "lower is better"),
        ("val_rec_loss", "Reconstruction Loss", "lower is better"),
        ("val_codebook_util", "Codebook Utilization (%)", "higher is better"),
    ]
    
    # Load all data at epoch 1 (common epoch)
    target_epoch = 1
    
    model_data = {}  # model -> {size_num: metrics}
    
    for model_name, (model_dir, prefix) in models_config.items():
        model_data[model_name] = {}
        
        for size_name, (size_num, size_suffix) in codebook_sizes.items():
            exp_dir = results_dir / model_dir / f"{prefix}{size_suffix}"
            if not exp_dir.exists():
                continue
            
            metrics = load_metrics_at_epoch(exp_dir, target_epoch)
            if metrics:
                model_data[model_name][size_num] = metrics
                print(f"Loaded {model_name} {size_name}")
    
    # Colors and markers for each model
    model_styles = {
        "LMB": {"color": "#1f77b4", "marker": "o"},
        "FSQ": {"color": "#2ca02c", "marker": "s"},
        "LFQ": {"color": "#d62728", "marker": "^"},
        "VQ": {"color": "#7f7f7f", "marker": "D"},
        "SIM_VQ": {"color": "#9467bd", "marker": "v"},
    }
    
    # X-axis values (codebook sizes)
    x_values = sorted([size_num for _, (size_num, _) in codebook_sizes.items()])
    x_labels = ["4K", "8K", "16K", "32K", "65K"]
    
    # Create one plot per metric
    for metric_key, metric_name, direction in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in models_config.keys():
            if model_name not in model_data:
                continue
            
            sizes = []
            values = []
            
            for size_num in x_values:
                if size_num in model_data[model_name]:
                    val = model_data[model_name][size_num].get(metric_key)
                    if val is not None:
                        sizes.append(size_num)
                        values.append(val)
            
            if sizes:
                style = model_styles[model_name]
                ax.plot(sizes, values, 
                       marker=style["marker"], 
                       color=style["color"],
                       linewidth=2.5,
                       markersize=10,
                       label=model_name)
                
                # Add value annotations
                for x, y in zip(sizes, values):
                    ax.annotate(f'{y:.1f}' if y > 10 else f'{y:.3f}',
                               xy=(x, y),
                               xytext=(5, 5),
                               textcoords='offset points',
                               fontsize=8,
                               color=style["color"])
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Codebook Size', fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(f'{metric_name} vs Codebook Size (Epoch {target_epoch})\n({direction})', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"codebook_scaling_{metric_key.replace('val_', '')}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    
    # Also create a combined 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name, direction) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for model_name in models_config.keys():
            if model_name not in model_data:
                continue
            
            sizes = []
            values = []
            
            for size_num in x_values:
                if size_num in model_data[model_name]:
                    val = model_data[model_name][size_num].get(metric_key)
                    if val is not None:
                        sizes.append(size_num)
                        values.append(val)
            
            if sizes:
                style = model_styles[model_name]
                ax.plot(sizes, values, 
                       marker=style["marker"], 
                       color=style["color"],
                       linewidth=2,
                       markersize=8,
                       label=model_name)
        
        ax.set_xscale('log', base=2)
        ax.set_xticks(x_values)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Codebook Size')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name}\n({direction})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Codebook Size Scaling Comparison (Epoch {target_epoch})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    output_path = output_dir / "codebook_scaling_all_metrics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

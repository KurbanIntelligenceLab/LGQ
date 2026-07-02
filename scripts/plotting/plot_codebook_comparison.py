#!/usr/bin/env python3
"""
Plot codebook size comparison across all models (LMB, FSQ, LFQ, VQ, SIM_VQ).
"""

import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def plot_codebook_comparison_bar(
    models_data: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    epoch: int
):
    """Plot bar chart comparing codebook sizes across models at a specific epoch."""
    
    metrics_to_plot = [
        ("val_rfid", "rFID (↓)", False),
        ("val_psnr", "PSNR (dB) (↑)", True),
        ("val_ssim", "SSIM (↑)", True),
        ("val_lpips", "LPIPS (↓)", False),
        ("val_rec_loss", "Rec Loss (↓)", False),
        ("val_codebook_util", "Codebook Util (%) (↑)", True),
    ]
    
    codebook_sizes = ["4K", "8K", "16K", "32K", "65K"]
    models = list(models_data.keys())
    
    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # Colors for each model
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    model_colors = {model: colors[i] for i, model in enumerate(models)}
    
    x = np.arange(len(codebook_sizes))
    width = 0.15
    
    for idx, (metric_key, metric_name, higher_is_better) in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        for i, model in enumerate(models):
            values = []
            for size in codebook_sizes:
                if size in models_data[model] and models_data[model][size]:
                    val = models_data[model][size].get(metric_key)
                    values.append(val if val is not None else np.nan)
                else:
                    values.append(np.nan)
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=model_colors[model])
            
            # Add value labels on top of bars
            for j, v in enumerate(values):
                if not np.isnan(v):
                    ax.annotate(f'{v:.1f}' if v > 10 else f'{v:.2f}',
                               xy=(x[j] + offset, v),
                               ha='center', va='bottom',
                               fontsize=6, rotation=90)
        
        ax.set_xlabel('Codebook Size')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels(codebook_sizes)
        ax.legend(fontsize=8, loc='best')
    
    fig.suptitle(f'Codebook Size Comparison Across Models (Epoch {epoch})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_codebook_comparison_lines(
    all_models_epochs: Dict[str, Dict[str, Dict[int, Dict[str, float]]]],
    output_path: Path
):
    """Plot line chart comparing codebook sizes across models over epochs."""
    
    metrics_to_plot = [
        ("val_rfid", "rFID (↓)", False),
        ("val_psnr", "PSNR (dB) (↑)", True),
        ("val_ssim", "SSIM (↑)", True),
        ("val_lpips", "LPIPS (↓)", False),
        ("val_rec_loss", "Rec Loss (↓)", False),
        ("val_codebook_util", "Codebook Util (%) (↑)", True),
    ]
    
    codebook_sizes = ["4K", "8K", "16K", "32K", "65K"]
    models = list(all_models_epochs.keys())
    
    # Create one figure per codebook size
    for size in codebook_sizes:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        axes = axes.flatten()
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        
        has_data = False
        for idx, (metric_key, metric_name, higher_is_better) in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            for model, color in zip(models, colors):
                if size not in all_models_epochs[model]:
                    continue
                    
                epochs_data = all_models_epochs[model][size]
                if not epochs_data:
                    continue
                
                epochs = sorted(epochs_data.keys())
                values = []
                valid_epochs = []
                
                for epoch in epochs:
                    val = epochs_data[epoch].get(metric_key)
                    if val is not None:
                        values.append(val)
                        valid_epochs.append(epoch)
                
                if valid_epochs:
                    has_data = True
                    ax.plot(valid_epochs, values, 'o-', label=model, color=color, 
                           markersize=6, linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name)
            ax.set_title(metric_name)
            ax.legend(fontsize=8, loc='best')
        
        if has_data:
            fig.suptitle(f'Model Comparison - Codebook Size {size}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            out_file = output_path.parent / f"codebook_{size.lower()}_models_comparison.png"
            plt.savefig(out_file, dpi=150, bbox_inches='tight')
            print(f"Saved: {out_file}")
        plt.close()


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
        "4K": "4k",
        "8K": "8k", 
        "16K": "16k",
        "32K": "32k",
        "65K": "65k",
    }
    
    # Load all data
    all_models_epochs = {}  # model -> size -> epoch -> metrics
    
    for model_name, (model_dir, prefix) in models_config.items():
        all_models_epochs[model_name] = {}
        
        for size_name, size_suffix in codebook_sizes.items():
            exp_dir = results_dir / model_dir / f"{prefix}{size_suffix}"
            if not exp_dir.exists():
                continue
            
            epochs_data = load_all_epochs_metrics(exp_dir)
            if epochs_data:
                all_models_epochs[model_name][size_name] = epochs_data
                print(f"Loaded {model_name} {size_name}: {len(epochs_data)} epochs")
    
    # Find common epoch for bar chart comparison
    all_epochs = set()
    for model_data in all_models_epochs.values():
        for size_data in model_data.values():
            if size_data:
                all_epochs.update(size_data.keys())
    
    if all_epochs:
        # Use epoch 1 as common comparison point (most experiments have it)
        common_epoch = 1
        
        # Prepare data for bar chart
        models_data = {}
        for model_name in all_models_epochs:
            models_data[model_name] = {}
            for size_name in codebook_sizes:
                if size_name in all_models_epochs[model_name]:
                    epochs_data = all_models_epochs[model_name][size_name]
                    if common_epoch in epochs_data:
                        models_data[model_name][size_name] = epochs_data[common_epoch]
        
        # Plot bar chart
        output_path = output_dir / "codebook_size_all_models_comparison.png"
        plot_codebook_comparison_bar(models_data, output_path, common_epoch)
        
        # Plot line charts per codebook size
        plot_codebook_comparison_lines(all_models_epochs, output_dir / "dummy.png")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

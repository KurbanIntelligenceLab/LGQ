#!/usr/bin/env python3
"""
Create comparison plots for FSQ, LMB flattened, and LMB per-channel fair (FSQ-style).
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    """Load evaluation metrics from a run directory."""
    eval_path = run_dir / "eval_metrics.csv"
    if not eval_path.exists():
        return [], {}
    
    epochs = []
    metrics = {
        'rfid': [],
        'psnr': [],
        'ssim': [],
        'lpips': [],
        'rec_loss': [],
        'active_codes': [],
        'codebook_util': [],
    }
    
    with open(eval_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row['epoch'])
                epochs.append(epoch)
                metrics['rfid'].append(float(row['val_rfid']) if row.get('val_rfid') and row['val_rfid'] != 'nan' else None)
                metrics['psnr'].append(float(row['val_psnr']))
                metrics['ssim'].append(float(row['val_ssim']))
                metrics['lpips'].append(float(row['val_lpips']))
                metrics['rec_loss'].append(float(row['val_rec_loss']))
                metrics['active_codes'].append(int(row['val_active_codes']))
                metrics['codebook_util'].append(float(row['val_codebook_util']))
            except (ValueError, KeyError) as e:
                continue
    
    return epochs, metrics


def create_comparison_plots():
    """Create comparison plots for FSQ, LMB flattened, and LMB per-channel fair."""
    
    # Get the script directory and project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Define experiment directories (relative to project root)
    experiments = {
        'FSQ (cb16k)': project_root / "results/fsq/fsq_cb16k",
        'LMB Flattened': project_root / "results/lmb/lmb_ablation_flattened",
        'LMB Per-channel Fair': project_root / "results/lmb/lmb_ablation_perchannel_fair",
    }
    
    # Load data for each experiment
    data = {}
    for name, run_dir in experiments.items():
        epochs, metrics = load_metrics(run_dir)
        if epochs:
            data[name] = {
                'epochs': epochs,
                'metrics': metrics
            }
            print(f"Loaded {name}: {len(epochs)} epochs")
        else:
            print(f"Warning: No data found for {name}")
    
    if not data:
        print("No data found for any experiment!")
        return
    
    # Define colors and markers
    colors = {
        'FSQ (cb16k)': '#1f77b4',
        'LMB Flattened': '#2ca02c',
        'LMB Per-channel Fair': '#d62728',
    }
    
    markers = {
        'FSQ (cb16k)': 'o',
        'LMB Flattened': 's',
        'LMB Per-channel Fair': '^',
    }
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('FSQ vs LMB Flattened vs LMB Per-channel Fair Comparison', fontsize=16, fontweight='bold')
    
    # Plot metrics
    metric_configs = [
        ('rfid', 'rFID (lower is better)', False, axes[0, 0]),
        ('psnr', 'PSNR (dB) (higher is better)', True, axes[0, 1]),
        ('ssim', 'SSIM (higher is better)', True, axes[0, 2]),
        ('lpips', 'LPIPS (lower is better)', False, axes[0, 3]),
        ('rec_loss', 'Rec Loss (lower is better)', False, axes[1, 0]),
        ('active_codes', 'Active Codes (higher is better)', True, axes[1, 1]),
        ('codebook_util', 'Codebook Util (%) (higher is better)', True, axes[1, 2]),
    ]
    
    for metric_key, ylabel, higher_better, ax in metric_configs:
        for name in data.keys():
            epochs = data[name]['epochs']
            values = data[name]['metrics'][metric_key]
            
            # Filter out None values
            valid_pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid_pairs:
                valid_epochs, valid_values = zip(*valid_pairs)
                ax.plot(
                    valid_epochs,
                    valid_values,
                    marker=markers[name],
                    markersize=8,
                    linewidth=2,
                    label=name,
                    color=colors[name],
                    alpha=0.8
                )
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(ylabel.split('(')[0].strip(), fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    # Add a summary table in the last subplot
    ax = axes[1, 3]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Model', 'Best rFID', 'Best PSNR', 'Best SSIM'])
    
    for name in data.keys():
        epochs = data[name]['epochs']
        rfid = data[name]['metrics']['rfid']
        psnr = data[name]['metrics']['psnr']
        ssim = data[name]['metrics']['ssim']
        
        # Find best values (filter None)
        valid_rfid = [(e, v) for e, v in zip(epochs, rfid) if v is not None]
        valid_psnr = [(e, v) for e, v in zip(epochs, psnr) if v is not None]
        valid_ssim = [(e, v) for e, v in zip(epochs, ssim) if v is not None]
        
        best_rfid = f"{min(valid_rfid, key=lambda x: x[1])[1]:.2f}" if valid_rfid else "N/A"
        best_psnr = f"{max(valid_psnr, key=lambda x: x[1])[1]:.2f}" if valid_psnr else "N/A"
        best_ssim = f"{max(valid_ssim, key=lambda x: x[1])[1]:.4f}" if valid_ssim else "N/A"
        
        table_data.append([name, best_rfid, best_psnr, best_ssim])
    
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.4, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary (Best Values)', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save plot
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    output_path = project_root / "results" / "plots" / "fsq_lmb_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    for name in data.keys():
        epochs = data[name]['epochs']
        print(f"\n{name}:")
        print(f"  Epochs: {min(epochs)}-{max(epochs)} ({len(epochs)} total)")
        rfid = data[name]['metrics']['rfid']
        psnr = data[name]['metrics']['psnr']
        ssim = data[name]['metrics']['ssim']
        valid_rfid = [v for v in rfid if v is not None]
        valid_psnr = [v for v in psnr if v is not None]
        valid_ssim = [v for v in ssim if v is not None]
        if valid_rfid:
            print(f"  Best rFID: {min(valid_rfid):.2f}")
        if valid_psnr:
            print(f"  Best PSNR: {max(valid_psnr):.2f} dB")
        if valid_ssim:
            print(f"  Best SSIM: {max(valid_ssim):.4f}")


if __name__ == "__main__":
    create_comparison_plots()

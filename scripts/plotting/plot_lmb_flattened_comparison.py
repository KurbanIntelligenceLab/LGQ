#!/usr/bin/env python3
"""
Create comprehensive plots comparing LMB Flattened, LMB Per-Channel Fair, and LMB Per-Channel Fair Flattened.
Shows both training metrics and evaluation metrics.
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple, Optional


def load_train_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    """Load training metrics from train_metrics.csv."""
    train_path = run_dir / "train_metrics.csv"
    if not train_path.exists():
        return [], {}
    
    epochs = []
    metrics = {
        'rec_loss': [],
        'vq_loss': [],
        'total_loss': [],
        'perplexity': [],
        'active_codes': [],
    }
    
    with open(train_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row['epoch'])
                epochs.append(epoch)
                metrics['rec_loss'].append(float(row.get('rec_loss', 0)))
                metrics['vq_loss'].append(float(row.get('vq_loss', 0)) if row.get('vq_loss') and row.get('vq_loss') != 'N/A' else None)
                metrics['total_loss'].append(float(row.get('total_loss', 0)))
                metrics['perplexity'].append(float(row.get('perplexity', 0)) if row.get('perplexity') and row.get('perplexity') != 'N/A' else None)
                metrics['active_codes'].append(int(row.get('active_codes', 0)) if row.get('active_codes') and row.get('active_codes') != 'N/A' else None)
            except (ValueError, KeyError) as e:
                continue
    
    return epochs, metrics


def load_eval_metrics(run_dir: Path) -> Tuple[List[int], Dict[str, List[float]]]:
    """Load evaluation metrics from eval_metrics.csv."""
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
        'perplexity': [],
    }
    
    with open(eval_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(row['epoch'])
                epochs.append(epoch)
                rfid_val = row.get('val_rfid') or row.get('val_rFID') or row.get('rfid')
                metrics['rfid'].append(float(rfid_val) if rfid_val and rfid_val.lower() != 'nan' else None)
                metrics['psnr'].append(float(row.get('val_psnr') or row.get('val_PSNR') or row.get('psnr', 0)))
                metrics['ssim'].append(float(row.get('val_ssim') or row.get('val_SSIM') or row.get('ssim', 0)))
                metrics['lpips'].append(float(row.get('val_lpips') or row.get('val_LPIPS') or row.get('lpips', 0)))
                metrics['rec_loss'].append(float(row.get('val_rec_loss') or row.get('rec_loss', 0)))
                metrics['active_codes'].append(int(row.get('val_active_codes') or row.get('active_codes', 0)))
                metrics['codebook_util'].append(float(row.get('val_codebook_util') or row.get('codebook_util', 0)))
                metrics['perplexity'].append(float(row.get('val_perplexity') or row.get('perplexity', 0)) if row.get('val_perplexity') and row.get('val_perplexity') != 'N/A' else None)
            except (ValueError, KeyError) as e:
                continue
    
    return epochs, metrics


def create_comparison_plots():
    """Create comprehensive comparison plots for the three LMB variants."""
    
    # Define experiment directories
    experiments = {
        'LMB Flattened\n[16384, 128]': Path("results/lmb/lmb_ablation_flattened"),
        'LMB Per-Ch Fair\n[4, 16] scalar': Path("results/lmb/lmb_ablation_perchannel_fair"),
        'LMB Per-Ch Fair Flattened\n[16384, 4]': Path("results/lmb/lmb_ablation_perchannel_fair_flattened"),
    }
    
    # Load data for each experiment
    train_data = {}
    eval_data = {}
    
    for name, run_dir in experiments.items():
        train_epochs, train_metrics = load_train_metrics(run_dir)
        eval_epochs, eval_metrics = load_eval_metrics(run_dir)
        
        if train_epochs:
            train_data[name] = {
                'epochs': train_epochs,
                'metrics': train_metrics
            }
            print(f"Loaded training data for {name}: {len(train_epochs)} epochs")
        
        if eval_epochs:
            eval_data[name] = {
                'epochs': eval_epochs,
                'metrics': eval_metrics
            }
            print(f"Loaded eval data for {name}: {len(eval_epochs)} epochs")
    
    if not train_data and not eval_data:
        print("No data found for any experiment!")
        return
    
    # Define colors and markers
    colors = {
        'LMB Flattened\n[16384, 128]': '#2ca02c',
        'LMB Per-Ch Fair\n[4, 16] scalar': '#d62728',
        'LMB Per-Ch Fair Flattened\n[16384, 4]': '#ff7f0e',
    }
    
    markers = {
        'LMB Flattened\n[16384, 128]': 'o',
        'LMB Per-Ch Fair\n[4, 16] scalar': 's',
        'LMB Per-Ch Fair Flattened\n[16384, 4]': '^',
    }
    
    # Create figure with subplots - Training metrics on top, Eval metrics on bottom
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('LMB Variants Comparison: Training & Evaluation Metrics', fontsize=18, fontweight='bold', y=0.995)
    
    # ===== TRAINING METRICS (Top row) =====
    train_axes = [
        fig.add_subplot(gs[0, 0]),  # Rec Loss
        fig.add_subplot(gs[0, 1]),  # VQ Loss
        fig.add_subplot(gs[0, 2]),  # Total Loss
        fig.add_subplot(gs[0, 3]),  # Perplexity
    ]
    
    train_configs = [
        ('rec_loss', 'Rec Loss', False, train_axes[0]),
        ('vq_loss', 'VQ Loss', False, train_axes[1]),
        ('total_loss', 'Total Loss', False, train_axes[2]),
        ('perplexity', 'Perplexity', True, train_axes[3]),
    ]
    
    for metric_key, ylabel, higher_better, ax in train_configs:
        for name in train_data.keys():
            epochs = train_data[name]['epochs']
            values = train_data[name]['metrics'][metric_key]
            
            # Filter out None values
            valid_pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid_pairs:
                valid_epochs, valid_values = zip(*valid_pairs)
                ax.plot(
                    valid_epochs,
                    valid_values,
                    marker=markers[name],
                    markersize=6,
                    linewidth=2,
                    label=name,
                    color=colors[name],
                    alpha=0.8,
                    markevery=max(1, len(valid_epochs)//20)  # Show markers every N points
                )
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if metric_key == 'rec_loss':  # Only show legend on first plot
            ax.legend(fontsize=9, loc='best')
    
    # ===== EVALUATION METRICS (Middle row) =====
    eval_axes = [
        fig.add_subplot(gs[1, 0]),  # rFID
        fig.add_subplot(gs[1, 1]),  # PSNR
        fig.add_subplot(gs[1, 2]),  # SSIM
        fig.add_subplot(gs[1, 3]),  # LPIPS
    ]
    
    eval_configs = [
        ('rfid', 'rFID (lower is better)', False, eval_axes[0]),
        ('psnr', 'PSNR (dB)', True, eval_axes[1]),
        ('ssim', 'SSIM', True, eval_axes[2]),
        ('lpips', 'LPIPS (lower is better)', False, eval_axes[3]),
    ]
    
    for metric_key, ylabel, higher_better, ax in eval_configs:
        for name in eval_data.keys():
            epochs = eval_data[name]['epochs']
            values = eval_data[name]['metrics'][metric_key]
            
            # Filter out None values
            valid_pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid_pairs:
                valid_epochs, valid_values = zip(*valid_pairs)
                ax.plot(
                    valid_epochs,
                    valid_values,
                    marker=markers[name],
                    markersize=8,
                    linewidth=2.5,
                    label=name,
                    color=colors[name],
                    alpha=0.8
                )
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel.split('(')[0].strip(), fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if metric_key == 'rfid':  # Only show legend on first plot
            ax.legend(fontsize=9, loc='best')
    
    # ===== CODEBOOK METRICS (Bottom row) =====
    codebook_axes = [
        fig.add_subplot(gs[2, 0]),  # Codebook Util
        fig.add_subplot(gs[2, 1]),  # Active Codes
        fig.add_subplot(gs[2, 2]),  # Eval Perplexity
        fig.add_subplot(gs[2, 3]),  # Summary table
    ]
    
    codebook_configs = [
        ('codebook_util', 'Codebook Util (%)', True, codebook_axes[0]),
        ('active_codes', 'Active Codes', True, codebook_axes[1]),
        ('perplexity', 'Perplexity (eval)', True, codebook_axes[2]),
    ]
    
    for metric_key, ylabel, higher_better, ax in codebook_configs:
        for name in eval_data.keys():
            epochs = eval_data[name]['epochs']
            values = eval_data[name]['metrics'][metric_key]
            
            # Filter out None values
            valid_pairs = [(e, v) for e, v in zip(epochs, values) if v is not None]
            if valid_pairs:
                valid_epochs, valid_values = zip(*valid_pairs)
                ax.plot(
                    valid_epochs,
                    valid_values,
                    marker=markers[name],
                    markersize=8,
                    linewidth=2.5,
                    label=name,
                    color=colors[name],
                    alpha=0.8
                )
        
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if metric_key == 'codebook_util':  # Only show legend on first plot
            ax.legend(fontsize=9, loc='best')
    
    # ===== SUMMARY TABLE =====
    ax = codebook_axes[3]
    ax.axis('off')
    
    # Create summary table
    table_data = []
    table_data.append(['Model', 'Best rFID', 'Best PSNR', 'Best SSIM', 'Codebook Util'])
    
    for name in eval_data.keys():
        epochs = eval_data[name]['epochs']
        rfid = eval_data[name]['metrics']['rfid']
        psnr = eval_data[name]['metrics']['psnr']
        ssim = eval_data[name]['metrics']['ssim']
        util = eval_data[name]['metrics']['codebook_util']
        
        # Find best values (filter None)
        valid_rfid = [(e, v) for e, v in zip(epochs, rfid) if v is not None]
        valid_psnr = [(e, v) for e, v in zip(epochs, psnr) if v is not None]
        valid_ssim = [(e, v) for e, v in zip(epochs, ssim) if v is not None]
        valid_util = [v for v in util if v is not None]
        
        best_rfid = f"{min(valid_rfid, key=lambda x: x[1])[1]:.2f}" if valid_rfid else "N/A"
        best_psnr = f"{max(valid_psnr, key=lambda x: x[1])[1]:.2f}" if valid_psnr else "N/A"
        best_ssim = f"{max(valid_ssim, key=lambda x: x[1])[1]:.4f}" if valid_ssim else "N/A"
        avg_util = f"{np.mean(valid_util):.1f}%" if valid_util else "N/A"
        
        # Shorten name for table
        short_name = name.replace('\n', ' ')
        table_data.append([short_name, best_rfid, best_psnr, best_ssim, avg_util])
    
    table = ax.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.15, 0.15, 0.15, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Summary (Best Values)', fontsize=12, fontweight='bold', pad=20)
    
    # Save plot
    output_path = Path("results") / "plots" / "lmb_flattened_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("LMB Variants Comparison Summary")
    print("="*80)
    for name in eval_data.keys():
        epochs = eval_data[name]['epochs']
        print(f"\n{name}:")
        print(f"  Epochs: {min(epochs)}-{max(epochs)} ({len(epochs)} total)")
        rfid = eval_data[name]['metrics']['rfid']
        psnr = eval_data[name]['metrics']['psnr']
        ssim = eval_data[name]['metrics']['ssim']
        util = eval_data[name]['metrics']['codebook_util']
        valid_rfid = [v for v in rfid if v is not None]
        valid_psnr = [v for v in psnr if v is not None]
        valid_ssim = [v for v in ssim if v is not None]
        valid_util = [v for v in util if v is not None]
        if valid_rfid:
            print(f"  Best rFID: {min(valid_rfid):.2f}")
        if valid_psnr:
            print(f"  Best PSNR: {max(valid_psnr):.2f} dB")
        if valid_ssim:
            print(f"  Best SSIM: {max(valid_ssim):.4f}")
        if valid_util:
            print(f"  Avg Codebook Util: {np.mean(valid_util):.1f}%")


if __name__ == "__main__":
    create_comparison_plots()

#!/usr/bin/env python3
"""
Create plots for LMB flattened model metrics.
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def create_plots():
    """Create visualization plots for the metrics."""
    run_dir = Path("results/lmb/lmb_nb16384_tau1.0-0.1_bs64_lr3e-4_dim128_20260119_214749_6c81")
    eval_path = run_dir / "eval_metrics.csv"
    train_path = run_dir / "train_metrics.csv"
    
    # Load eval metrics
    epochs_eval = []
    rfid = []
    psnr = []
    ssim = []
    lpips = []
    rec_loss_eval = []
    active_codes_eval = []
    
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs_eval.append(int(row['epoch']))
                rfid.append(float(row['val_rfid']))
                psnr.append(float(row['val_psnr']))
                ssim.append(float(row['val_ssim']))
                lpips.append(float(row['val_lpips']))
                rec_loss_eval.append(float(row['val_rec_loss']))
                active_codes_eval.append(int(row['val_active_codes']))
    
    # Load train metrics (for training curves)
    epochs_train = []
    rec_loss_train = []
    active_codes_train = []
    perplexity_train = []
    
    if train_path.exists():
        with open(train_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epoch = int(row['epoch'])
                epochs_train.append(epoch)
                rec_loss_train.append(float(row['rec_loss']))
                active_codes_train.append(int(row['active_codes']))
                perplexity_train.append(float(row['perplexity']))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LMB Flattened Model - Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: rFID (eval only)
    if epochs_eval:
        axes[0, 0].plot(epochs_eval, rfid, 'o-', linewidth=2, markersize=10, color='#1f77b4', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('rFID (lower is better)')
        axes[0, 0].set_title('Reconstruction FID')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
    
    # Plot 2: PSNR (eval only)
    if epochs_eval:
        axes[0, 1].plot(epochs_eval, psnr, 'o-', linewidth=2, markersize=10, color='#ff7f0e', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('PSNR (higher is better)')
        axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
    
    # Plot 3: SSIM (eval only)
    if epochs_eval:
        axes[0, 2].plot(epochs_eval, ssim, 'o-', linewidth=2, markersize=10, color='#2ca02c', label='Validation')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('SSIM (higher is better)')
        axes[0, 2].set_title('Structural Similarity Index')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].legend()
    
    # Plot 4: LPIPS (eval only)
    if epochs_eval:
        axes[1, 0].plot(epochs_eval, lpips, 'o-', linewidth=2, markersize=10, color='#d62728', label='Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('LPIPS (lower is better)')
        axes[1, 0].set_title('Learned Perceptual Image Patch Similarity')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
    
    # Plot 5: Rec Loss (both train and eval)
    if epochs_train:
        # Smooth training curve
        from collections import defaultdict
        epoch_to_loss = defaultdict(list)
        for e, l in zip(epochs_train, rec_loss_train):
            epoch_to_loss[e].append(l)
        epochs_smooth = sorted(epoch_to_loss.keys())
        loss_smooth = [np.mean(epoch_to_loss[e]) for e in epochs_smooth]
        axes[1, 1].plot(epochs_smooth, loss_smooth, '-', linewidth=1.5, alpha=0.7, color='#9467bd', label='Training (avg)')
    
    if epochs_eval:
        axes[1, 1].plot(epochs_eval, rec_loss_eval, 'o-', linewidth=2, markersize=10, color='#9467bd', label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reconstruction Loss')
    axes[1, 1].set_title('Reconstruction Loss')
    axes[1, 1].grid(True, alpha=0.3)
    if epochs_train or epochs_eval:
        axes[1, 1].legend()
    
    # Plot 6: Active Codes (both train and eval)
    if epochs_train:
        epoch_to_codes = defaultdict(list)
        for e, c in zip(epochs_train, active_codes_train):
            epoch_to_codes[e].append(c)
        epochs_smooth = sorted(epoch_to_codes.keys())
        codes_smooth = [np.mean(epoch_to_codes[e]) for e in epochs_smooth]
        axes[1, 2].plot(epochs_smooth, codes_smooth, '-', linewidth=1.5, alpha=0.7, color='#8c564b', label='Training (avg)')
    
    if epochs_eval:
        axes[1, 2].plot(epochs_eval, active_codes_eval, 'o-', linewidth=2, markersize=10, color='#8c564b', label='Validation')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Active Codes')
    axes[1, 2].set_title('Codebook Utilization')
    axes[1, 2].grid(True, alpha=0.3)
    if epochs_train or epochs_eval:
        axes[1, 2].legend()
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("plots") / "lmb_flattened_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()
    
    # Print summary
    print("\n" + "="*80)
    print("LMB Flattened Model - Evaluation Summary")
    print("="*80)
    if epochs_eval:
        for i, epoch in enumerate(epochs_eval):
            print(f"\nEpoch {epoch}:")
            print(f"  rFID: {rfid[i]:.2f}")
            print(f"  PSNR: {psnr[i]:.2f}")
            print(f"  SSIM: {ssim[i]:.4f}")
            print(f"  LPIPS: {lpips[i]:.4f}")
            print(f"  Rec Loss: {rec_loss_eval[i]:.4f}")
            print(f"  Active Codes: {active_codes_eval[i]} ({100*active_codes_eval[i]/16384:.2f}%)")
    else:
        print("\nNo evaluation metrics found. Run evaluation first.")

if __name__ == "__main__":
    create_plots()

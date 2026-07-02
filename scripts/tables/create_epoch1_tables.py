#!/usr/bin/env python3
"""
Create tables comparing first epoch metrics for all models.
"""

import csv
import json
from pathlib import Path

def load_first_epoch_metrics(eval_path):
    """Load first epoch metrics from eval_metrics.csv."""
    if not eval_path.exists():
        return None
    
    with open(eval_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if rows:
            return rows[0]  # First epoch
    return None

def create_all_models_table():
    """Create table with first epoch metrics for all models."""
    model_paths = {
        'VQ': 'results/vq/vq_fair_v2/eval_metrics.csv',
        'FSQ': 'results/fsq/fsq_fair_v2/eval_metrics.csv',
        'SimVQ': 'results/sim_vq/sim_vq_fair_v2/eval_metrics.csv',
        'LMB (Flattened)': 'results/lmb/lmb_nb16384_tau1.0-0.1_bs64_lr3e-4_dim128_20260119_214749_6c81/eval_metrics.csv',
    }
    
    results = {}
    for model_name, path in model_paths.items():
        metrics = load_first_epoch_metrics(Path(path))
        if metrics:
            results[model_name] = metrics
    
    if not results:
        print("No evaluation metrics found!")
        return
    
    # Create formatted table
    print("=" * 100)
    print("First Epoch Comparison - All Models")
    print("=" * 100)
    print()
    print(f"{'Model':<15} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'Active Codes':<15} {'Util %':<8}")
    print("-" * 100)
    
    for model_name in ['VQ', 'FSQ', 'SimVQ', 'LMB (Flattened)']:
        if model_name in results:
            m = results[model_name]
            rfid = float(m.get('val_rfid', 0))
            psnr = float(m.get('val_psnr', 0))
            ssim = float(m.get('val_ssim', 0))
            lpips = float(m.get('val_lpips', 0))
            rec_loss = float(m.get('val_rec_loss', 0))
            active_codes = int(m.get('val_active_codes', 0))
            util = float(m.get('val_codebook_util', 0))
            
            print(f"{model_name:<15} {rfid:<10.2f} {psnr:<8.2f} {ssim:<8.4f} {lpips:<8.4f} {rec_loss:<10.4f} {active_codes:<15} {util:<8.2f}")
    
    print()
    print("=" * 100)
    
    # Save to file
    output_path = Path("results") / "first_epoch_all_models.txt"
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("First Epoch Comparison - All Models\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Model':<15} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'Active Codes':<15} {'Util %':<8}\n")
        f.write("-" * 100 + "\n")
        
        for model_name in ['VQ', 'FSQ', 'SimVQ', 'LMB (Flattened)']:
            if model_name in results:
                m = results[model_name]
                rfid = float(m.get('val_rfid', 0))
                psnr = float(m.get('val_psnr', 0))
                ssim = float(m.get('val_ssim', 0))
                lpips = float(m.get('val_lpips', 0))
                rec_loss = float(m.get('val_rec_loss', 0))
                active_codes = int(m.get('val_active_codes', 0))
                util = float(m.get('val_codebook_util', 0))
                
                f.write(f"{model_name:<15} {rfid:<10.2f} {psnr:<8.2f} {ssim:<8.4f} {lpips:<8.4f} {rec_loss:<10.4f} {active_codes:<15} {util:<8.2f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"Table saved to {output_path}")

def create_lmb_comparison_table():
    """Create comparison table for LMB_fair_5ch7bins vs LMB flattened."""
    paths = {
        'LMB Fair 5ch7bins': 'results/lmb/lmb_fair_5ch7bins/eval_metrics.csv',
        'LMB Flattened': 'results/lmb/lmb_nb16384_tau1.0-0.1_bs64_lr3e-4_dim128_20260119_214749_6c81/eval_metrics.csv',
    }
    
    results = {}
    for model_name, path in paths.items():
        metrics = load_first_epoch_metrics(Path(path))
        if metrics:
            results[model_name] = metrics
    
    if not results:
        print("No LMB evaluation metrics found!")
        return
    
    # Create formatted table
    print("\n" + "=" * 100)
    print("First Epoch Comparison - LMB Variants")
    print("=" * 100)
    print()
    print(f"{'Model':<20} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'VQ Loss':<10} {'Active Codes':<15} {'Util %':<8} {'Perplexity':<12}")
    print("-" * 100)
    
    for model_name in ['LMB Fair 5ch7bins', 'LMB Flattened']:
        if model_name in results:
            m = results[model_name]
            rfid = float(m.get('val_rfid', 0))
            psnr = float(m.get('val_psnr', 0))
            ssim = float(m.get('val_ssim', 0))
            lpips = float(m.get('val_lpips', 0))
            rec_loss = float(m.get('val_rec_loss', 0))
            vq_loss = float(m.get('val_vq_loss', 0))
            active_codes = int(m.get('val_active_codes', 0))
            util = float(m.get('val_codebook_util', 0))
            perplexity = float(m.get('val_perplexity', 0))
            
            print(f"{model_name:<20} {rfid:<10.2f} {psnr:<8.2f} {ssim:<8.4f} {lpips:<8.4f} {rec_loss:<10.4f} {vq_loss:<10.4f} {active_codes:<15} {util:<8.2f} {perplexity:<12.2f}")
    
    print()
    print("=" * 100)
    
    # Calculate differences
    if 'LMB Fair 5ch7bins' in results and 'LMB Flattened' in results:
        fair = results['LMB Fair 5ch7bins']
        flat = results['LMB Flattened']
        
        print("\nDifferences (Flattened - Fair):")
        print("-" * 100)
        print(f"rFID:     {float(flat.get('val_rfid', 0)) - float(fair.get('val_rfid', 0)):+.2f}")
        print(f"PSNR:     {float(flat.get('val_psnr', 0)) - float(fair.get('val_psnr', 0)):+.2f}")
        print(f"SSIM:     {float(flat.get('val_ssim', 0)) - float(fair.get('val_ssim', 0)):+.4f}")
        print(f"LPIPS:    {float(flat.get('val_lpips', 0)) - float(fair.get('val_lpips', 0)):+.4f}")
        print(f"Rec Loss: {float(flat.get('val_rec_loss', 0)) - float(fair.get('val_rec_loss', 0)):+.4f}")
        print(f"Active Codes: {int(flat.get('val_active_codes', 0)) - int(fair.get('val_active_codes', 0)):+d}")
        print(f"Util %:   {float(flat.get('val_codebook_util', 0)) - float(fair.get('val_codebook_util', 0)):+.2f}")
        print()
    
    # Save to file
    output_path = Path("results") / "lmb_first_epoch_comparison.txt"
    with open(output_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("First Epoch Comparison - LMB Variants\n")
        f.write("=" * 100 + "\n\n")
        f.write(f"{'Model':<20} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'VQ Loss':<10} {'Active Codes':<15} {'Util %':<8} {'Perplexity':<12}\n")
        f.write("-" * 100 + "\n")
        
        for model_name in ['LMB Fair 5ch7bins', 'LMB Flattened']:
            if model_name in results:
                m = results[model_name]
                rfid = float(m.get('val_rfid', 0))
                psnr = float(m.get('val_psnr', 0))
                ssim = float(m.get('val_ssim', 0))
                lpips = float(m.get('val_lpips', 0))
                rec_loss = float(m.get('val_rec_loss', 0))
                vq_loss = float(m.get('val_vq_loss', 0))
                active_codes = int(m.get('val_active_codes', 0))
                util = float(m.get('val_codebook_util', 0))
                perplexity = float(m.get('val_perplexity', 0))
                
                f.write(f"{model_name:<20} {rfid:<10.2f} {psnr:<8.2f} {ssim:<8.4f} {lpips:<8.4f} {rec_loss:<10.4f} {vq_loss:<10.4f} {active_codes:<15} {util:<8.2f} {perplexity:<12.2f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        
        if 'LMB Fair 5ch7bins' in results and 'LMB Flattened' in results:
            fair = results['LMB Fair 5ch7bins']
            flat = results['LMB Flattened']
            
            f.write("\nDifferences (Flattened - Fair):\n")
            f.write("-" * 100 + "\n")
            f.write(f"rFID:     {float(flat.get('val_rfid', 0)) - float(fair.get('val_rfid', 0)):+.2f}\n")
            f.write(f"PSNR:     {float(flat.get('val_psnr', 0)) - float(fair.get('val_psnr', 0)):+.2f}\n")
            f.write(f"SSIM:     {float(flat.get('val_ssim', 0)) - float(fair.get('val_ssim', 0)):+.4f}\n")
            f.write(f"LPIPS:    {float(flat.get('val_lpips', 0)) - float(fair.get('val_lpips', 0)):+.4f}\n")
            f.write(f"Rec Loss: {float(flat.get('val_rec_loss', 0)) - float(fair.get('val_rec_loss', 0)):+.4f}\n")
            f.write(f"Active Codes: {int(flat.get('val_active_codes', 0)) - int(fair.get('val_active_codes', 0)):+d}\n")
            f.write(f"Util %:   {float(flat.get('val_codebook_util', 0)) - float(fair.get('val_codebook_util', 0)):+.2f}\n")
    
    print(f"Table saved to {output_path}")

if __name__ == "__main__":
    create_all_models_table()
    create_lmb_comparison_table()

#!/usr/bin/env python3
"""
Compare codebook size ablation results across all models.
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

results_dir = Path('results')
models = ['fsq', 'lfq', 'sim_vq', 'vq', 'lmb']
codebook_sizes = ['4k', '8k', '16k', '32k', '65k']
codebook_map = {'4k': 4096, '8k': 8192, '16k': 16384, '32k': 32768, '65k': 65536}

def find_experiment(model, cb_size):
    model_dir = results_dir / model
    if not model_dir.exists():
        return None
    
    target_size = codebook_map[cb_size]
    
    # Collect all matching candidates
    candidates = []
    for exp_dir in model_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if f'cb{cb_size}' in exp_dir.name.lower() or f'cb{target_size}' in exp_dir.name.lower():
            config_path = exp_dir / 'config.json'
            eval_path = exp_dir / 'eval_metrics.csv'
            if config_path.exists() and eval_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Check if codebook size matches
                    matches = False
                    if model == 'fsq':
                        levels = config.get('levels', [])
                        if isinstance(levels, list):
                            total = 1
                            for l in levels:
                                total *= l
                            if total == target_size:
                                matches = True
                    elif model == 'lmb':
                        num_bins = config.get('num_bins', 0)
                        if num_bins == target_size:
                            matches = True
                    else:
                        cb_size_config = config.get('codebook_size', 0)
                        if cb_size_config == target_size:
                            matches = True
                    
                    if matches:
                        # Count epochs
                        epoch_count = 0
                        with open(eval_path, 'r') as ef:
                            reader = csv.DictReader(ef)
                            epoch_count = len(list(reader))
                        
                        # Prefer batch_size=32 for consistency within ablation study
                        # (most models have batch_size=32 experiments)
                        batch_size = config.get('batch_size', None)
                        is_preferred = (batch_size == 32)
                        
                        candidates.append((exp_dir, epoch_count, is_preferred, batch_size))
                except:
                    pass
    
    if not candidates:
        return None
    
    # Sort: prefer batch_size=32, then by epoch count
    candidates.sort(key=lambda x: (not x[2], -x[1]), reverse=False)
    return candidates[0][0]

def get_latest_metrics(exp_dir):
    eval_csv = exp_dir / 'eval_metrics.csv'
    if not eval_csv.exists():
        return None
    
    with open(eval_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            return None
        
        # Get last epoch
        last_row = rows[-1]
        return {
            'epoch': int(last_row.get('epoch', 0)),
            'rfid': float(last_row.get('val_rfid', 0)) if last_row.get('val_rfid') and last_row.get('val_rfid') != 'None' else None,
            'psnr': float(last_row.get('val_psnr', 0)) if last_row.get('val_psnr') and last_row.get('val_psnr') != 'None' else None,
            'ssim': float(last_row.get('val_ssim', 0)) if last_row.get('val_ssim') and last_row.get('val_ssim') != 'None' else None,
            'lpips': float(last_row.get('val_lpips', 0)) if last_row.get('val_lpips') and last_row.get('val_lpips') != 'None' else None,
            'rec_loss': float(last_row.get('val_rec_loss', 0)) if last_row.get('val_rec_loss') and last_row.get('val_rec_loss') != 'None' else None,
            'codebook_util': float(last_row.get('val_codebook_util', 0)) if last_row.get('val_codebook_util') and last_row.get('val_codebook_util') != 'None' else None,
            'active_codes': int(float(last_row.get('val_active_codes', 0))) if last_row.get('val_active_codes') and last_row.get('val_active_codes') != 'None' else None,
        }

# Collect all results
all_results = {}
for model in models:
    all_results[model] = {}
    for cb_size in codebook_sizes:
        exp_dir = find_experiment(model, cb_size)
        if exp_dir:
            metrics = get_latest_metrics(exp_dir)
            if metrics:
                all_results[model][cb_size] = metrics

# Print results
print('=' * 130)
print('CODEBOOK SIZE ABLATION RESULTS - COMPARISON ACROSS MODELS')
print('=' * 130)
print()

for cb_size in codebook_sizes:
    print(f'CODEBOOK SIZE: {codebook_map[cb_size]:,} ({cb_size.upper()})')
    print('-' * 130)
    header = f"{'Model':<10} {'Epoch':<8} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'Codebook Util':<15} {'Active Codes':<15}"
    print(header)
    print('-' * 130)
    
    for model in models:
        if cb_size in all_results[model]:
            m = all_results[model][cb_size]
            rfid_str = f"{m['rfid']:.2f}" if m['rfid'] is not None else "N/A"
            psnr_str = f"{m['psnr']:.2f}" if m['psnr'] is not None else "N/A"
            ssim_str = f"{m['ssim']:.4f}" if m['ssim'] is not None else "N/A"
            lpips_str = f"{m['lpips']:.4f}" if m['lpips'] is not None else "N/A"
            rec_str = f"{m['rec_loss']:.4f}" if m['rec_loss'] is not None else "N/A"
            util_str = f"{m['codebook_util']:.2f}%" if m['codebook_util'] is not None else "N/A"
            active_str = f"{m['active_codes']}" if m['active_codes'] is not None else "N/A"
            
            print(f"{model.upper():<10} {m['epoch']:<8} {rfid_str:<10} {psnr_str:<8} {ssim_str:<8} {lpips_str:<8} {rec_str:<10} {util_str:<15} {active_str:<15}")
        else:
            print(f"{model.upper():<10} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<15} {'N/A':<15}")
    print()

# Also create a model-by-model view
print('=' * 130)
print('MODEL-BY-MODEL CODEBOOK SIZE COMPARISON')
print('=' * 130)
print()

for model in models:
    print(f'{model.upper()} - Codebook Size Scaling')
    print('-' * 130)
    header = f"{'Codebook Size':<15} {'Epoch':<8} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'Codebook Util':<15} {'Active Codes':<15}"
    print(header)
    print('-' * 130)
    
    for cb_size in codebook_sizes:
        if cb_size in all_results[model]:
            m = all_results[model][cb_size]
            rfid_str = f"{m['rfid']:.2f}" if m['rfid'] is not None else "N/A"
            psnr_str = f"{m['psnr']:.2f}" if m['psnr'] is not None else "N/A"
            ssim_str = f"{m['ssim']:.4f}" if m['ssim'] is not None else "N/A"
            lpips_str = f"{m['lpips']:.4f}" if m['lpips'] is not None else "N/A"
            rec_str = f"{m['rec_loss']:.4f}" if m['rec_loss'] is not None else "N/A"
            util_str = f"{m['codebook_util']:.2f}%" if m['codebook_util'] is not None else "N/A"
            active_str = f"{m['active_codes']}" if m['active_codes'] is not None else "N/A"
            
            print(f"{cb_size.upper():<15} {m['epoch']:<8} {rfid_str:<10} {psnr_str:<8} {ssim_str:<8} {lpips_str:<8} {rec_str:<10} {util_str:<15} {active_str:<15}")
        else:
            print(f"{cb_size.upper():<15} {'N/A':<8} {'N/A':<10} {'N/A':<8} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'N/A':<15} {'N/A':<15}")
    print()

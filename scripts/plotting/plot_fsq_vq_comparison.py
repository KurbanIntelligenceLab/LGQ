#!/usr/bin/env python3
"""
Plot FSQ vs VQ vs LMB comparison across codebook sizes.
"""

import csv
import json
import math
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, Optional

plt.style.use('seaborn-v0_8-whitegrid')

def load_best_metrics(experiment_dir: Path) -> Optional[Dict[str, float]]:
    """Load best metrics from eval_metrics.csv (lowest rFID)."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return None
    epochs_metrics = []
    with open(eval_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('epoch'):
                continue
            try:
                epoch = int(row['epoch'])
                metrics = {'epoch': epoch}
                for key, value in row.items():
                    if key not in ['epoch', 'global_step']:
                        try:
                            if value and value.strip() and value != 'None':
                                metrics[key] = float(value)
                            else:
                                metrics[key] = None
                        except (ValueError, TypeError):
                            metrics[key] = None
                epochs_metrics.append(metrics)
            except (ValueError, KeyError):
                continue
    if not epochs_metrics:
        return None
    best_metrics = None
    best_rfid = float('inf')
    for metrics in epochs_metrics:
        rfid = metrics.get('val_rfid')
        if rfid is not None and rfid < best_rfid:
            best_rfid = rfid
            best_metrics = metrics
    return best_metrics if best_metrics else epochs_metrics[-1]

def calculate_compression_cost(active_codes: int, codebook_size: int, perplexity: Optional[float] = None) -> float:
    if perplexity is not None and perplexity > 0:
        return math.log2(perplexity)
    else:
        if active_codes > 0:
            return math.log2(active_codes)
        return 0.0

def find_experiments(model_type: str, codebook_sizes: list = None) -> Dict[int, Path]:
    results_dir = Path("results")
    model_dir = results_dir / model_type.lower()
    experiments = {}
    if not model_dir.exists():
        return experiments
    preferred_init = 'random' if model_type.lower() == 'lmb' else None
    for exp_dir in model_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        if not config_path.exists() or not (exp_dir / "eval_metrics.csv").exists():
            continue
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            cb_size = None
            if 'levels' in config:
                levels = config['levels']
                if isinstance(levels, list):
                    cb_size = int(np.prod(levels))
            elif 'codebook_size' in config:
                cb_size = int(config['codebook_size'])
            elif 'num_bins' in config:
                num_bins = int(config['num_bins'])
                flatten_channels = config.get('flatten_channels', False)
                perchannel_fair = config.get('perchannel_fair', False)
                if not flatten_channels and not perchannel_fair:
                    continue
                if perchannel_fair and 'lmb_levels' in config:
                    lmb_levels = config['lmb_levels']
                    if isinstance(lmb_levels, list):
                        cb_size = int(np.prod(lmb_levels))
                elif flatten_channels:
                    cb_size = num_bins
                else:
                    continue
            if cb_size:
                if codebook_sizes is None or cb_size in codebook_sizes:
                    try:
                        with open(exp_dir / "eval_metrics.csv", 'r') as f:
                            epoch_count = len(list(csv.DictReader(f)))
                        init_method = config.get('init_method', 'random')
                        if cb_size not in experiments:
                            experiments[cb_size] = (exp_dir, epoch_count, init_method)
                        else:
                            current_dir, current_epochs, current_init = experiments[cb_size]
                            if preferred_init and model_type.lower() == 'lmb':
                                if init_method == preferred_init and current_init != preferred_init:
                                    # Strongly prefer random init for consistency (even with fewer epochs)
                                    if epoch_count >= current_epochs * 0.3:  # Lower threshold
                                        experiments[cb_size] = (exp_dir, epoch_count, init_method)
                                elif init_method == preferred_init == current_init:
                                    # Both random: prefer more epochs
                                    if epoch_count > current_epochs:
                                        experiments[cb_size] = (exp_dir, epoch_count, init_method)
                                elif epoch_count > current_epochs * 3:  # Much more epochs needed to override
                                    experiments[cb_size] = (exp_dir, epoch_count, init_method)
                            else:
                                if epoch_count > current_epochs:
                                    experiments[cb_size] = (exp_dir, epoch_count, init_method)
                    except:
                        if cb_size not in experiments:
                            experiments[cb_size] = (exp_dir, 0, config.get('init_method', 'random'))
        except Exception:
            continue
    return {cb_size: exp_dir for cb_size, (exp_dir, _, _) in experiments.items()}

def main():
    results_dir = Path("results")
    output_dir = Path("results/plots")
    output_dir.mkdir(exist_ok=True)
    fsq_experiments = find_experiments("fsq", codebook_sizes=None)
    vq_experiments = find_experiments("vq", codebook_sizes=None)
    lmb_experiments = find_experiments("lmb", codebook_sizes=None)
    all_sizes = sorted(set(list(fsq_experiments.keys()) + list(vq_experiments.keys()) + list(lmb_experiments.keys())))
    common_sizes = sorted(set(fsq_experiments.keys()) & set(vq_experiments.keys()) & set(lmb_experiments.keys()))
    if common_sizes:
        codebook_sizes = common_sizes
    else:
        codebook_sizes = all_sizes
    print(f"Found {len(fsq_experiments)} FSQ experiments: {sorted(fsq_experiments.keys())}")
    print(f"Found {len(vq_experiments)} VQ experiments: {sorted(vq_experiments.keys())}")
    print(f"Found {len(lmb_experiments)} LMB experiments: {sorted(lmb_experiments.keys())}")
    print(f"Using codebook sizes: {codebook_sizes}")
    print("\nLMB experiment init methods:")
    for cb_size in sorted(lmb_experiments.keys()):
        config_path = lmb_experiments[cb_size] / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            init = config.get('init_method', 'random')
            print(f"  {cb_size}: {lmb_experiments[cb_size].name} - init={init}")
    fsq_data = {}
    vq_data = {}
    lmb_data = {}
    for cb_size in codebook_sizes:
        if cb_size in fsq_experiments:
            metrics = load_best_metrics(fsq_experiments[cb_size])
            if metrics:
                fsq_data[cb_size] = metrics
        if cb_size in vq_experiments:
            metrics = load_best_metrics(vq_experiments[cb_size])
            if metrics:
                vq_data[cb_size] = metrics
        if cb_size in lmb_experiments:
            metrics = load_best_metrics(lmb_experiments[cb_size])
            if metrics:
                lmb_data[cb_size] = metrics
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fsq_color = '#1f77b4'
    vq_color = '#e377c2'
    lmb_color = '#2ca02c'
    ax = axes[0]
    fsq_sizes = sorted([s for s in codebook_sizes if s in fsq_data])
    vq_sizes = sorted([s for s in codebook_sizes if s in vq_data])
    lmb_sizes = sorted([s for s in codebook_sizes if s in lmb_data])
    fsq_rfids = [fsq_data[s].get('val_rfid') for s in fsq_sizes]
    vq_rfids = [vq_data[s].get('val_rfid') for s in vq_sizes]
    lmb_rfids = [lmb_data[s].get('val_rfid') for s in lmb_sizes]
    if fsq_sizes:
        ax.plot(fsq_sizes, fsq_rfids, 'o-', color=fsq_color, linewidth=2.5, markersize=10, label='FSQ')
    if vq_sizes:
        ax.plot(vq_sizes, vq_rfids, 'o-', color=vq_color, linewidth=2.5, markersize=10, label='VQ')
    if lmb_sizes:
        ax.plot(lmb_sizes, lmb_rfids, 'o-', color=lmb_color, linewidth=2.5, markersize=10, label='LMB')
    ax.set_xscale('log', base=2)
    ax.set_xticks(codebook_sizes)
    ax.set_xticklabels([f'$2^{{{int(math.log2(s))}}}$' for s in codebook_sizes])
    ax.set_xlabel('Codebook Size', fontsize=12)
    ax.set_ylabel('FID', fontsize=12)
    ax.set_title('a) Reconstruction FID', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    fsq_active = [fsq_data[s].get('val_active_codes') or (fsq_data[s].get('val_codebook_util', 0) / 100.0 * s) for s in fsq_sizes]
    vq_active = [vq_data[s].get('val_active_codes') or (vq_data[s].get('val_codebook_util', 0) / 100.0 * s) for s in vq_sizes]
    lmb_active = [lmb_data[s].get('val_active_codes') or (lmb_data[s].get('val_codebook_util', 0) / 100.0 * s) for s in lmb_sizes]
    if fsq_sizes and fsq_active:
        ax.plot(fsq_sizes, fsq_active, 'o-', color=fsq_color, linewidth=2.5, markersize=10, label='FSQ')
    if vq_sizes and vq_active:
        ax.plot(vq_sizes, vq_active, 'o-', color=vq_color, linewidth=2.5, markersize=10, label='VQ')
    if lmb_sizes and lmb_active:
        ax.plot(lmb_sizes, lmb_active, 'o-', color=lmb_color, linewidth=2.5, markersize=10, label='LMB')
    max_line = codebook_sizes
    ax.plot(max_line, max_line, 'k-', linewidth=1.5, label='Maximum', alpha=0.7)
    ax.plot(max_line, [s * 0.5 for s in max_line], 'k--', linewidth=1, label='50%', alpha=0.5)
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_xticks(codebook_sizes)
    ax.set_xticklabels([f'$2^{{{int(math.log2(s))}}}$' for s in codebook_sizes])
    ax.set_xlabel('Codebook Size', fontsize=12)
    ax.set_ylabel('Codebook Usage', fontsize=12)
    ax.set_title('b) Codebook Usage', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax = axes[2]
    fsq_costs = [calculate_compression_cost(fsq_data[s].get('val_active_codes', s), s, fsq_data[s].get('val_perplexity')) for s in fsq_sizes]
    vq_costs = [calculate_compression_cost(vq_data[s].get('val_active_codes', s), s, vq_data[s].get('val_perplexity')) for s in vq_sizes]
    lmb_costs = [calculate_compression_cost(lmb_data[s].get('val_active_codes', s), s, lmb_data[s].get('val_perplexity')) for s in lmb_sizes]
    if fsq_sizes and fsq_costs:
        ax.plot(fsq_sizes, fsq_costs, 'o-', color=fsq_color, linewidth=2.5, markersize=10, label='FSQ')
    if vq_sizes and vq_costs:
        ax.plot(vq_sizes, vq_costs, 'o-', color=vq_color, linewidth=2.5, markersize=10, label='VQ')
    if lmb_sizes and lmb_costs:
        ax.plot(lmb_sizes, lmb_costs, 'o-', color=lmb_color, linewidth=2.5, markersize=10, label='LMB')
    uniform_costs = [math.log2(s) for s in codebook_sizes]
    ax.plot(codebook_sizes, uniform_costs, 'k-', linewidth=1.5, label='Uniform', alpha=0.7)
    ax.set_xscale('log', base=2)
    ax.set_xticks(codebook_sizes)
    ax.set_xticklabels([f'$2^{{{int(math.log2(s))}}}$' for s in codebook_sizes])
    ax.set_xlabel('Codebook Size', fontsize=12)
    ax.set_ylabel('Compression Cost [bits]', fontsize=12)
    ax.set_title('c) Compression Cost [bits]', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = output_dir / "fsq_vq_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_path}")
    print("\nDone!")

if __name__ == "__main__":
    main()

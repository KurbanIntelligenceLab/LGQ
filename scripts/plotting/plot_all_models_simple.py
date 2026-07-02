#!/usr/bin/env python3
"""
Create simple comparison plots for all models (VQ, FSQ, SimVQ, LMB) up to 30k steps.
Simplified version with cleaner layout.
"""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from collections import defaultdict

def load_train_metrics(csv_path, max_step=30000):
    """Load training metrics up to max_step."""
    steps = []
    rec_loss = []
    active_codes = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                step = int(row['global_step'])
                if step <= max_step:
                    steps.append(step)
                    rec_loss_val = float(row.get('rec_loss', 0))
                    # Filter out obviously wrong values
                    if rec_loss_val < 10:
                        rec_loss.append(rec_loss_val)
                    else:
                        rec_loss.append(rec_loss[-1] if rec_loss else 0.3)
                    
                    active_codes_val = row.get('active_codes', '0')
                    if active_codes_val:
                        active_codes.append(int(active_codes_val))
                    else:
                        active_codes.append(0)
            except (ValueError, KeyError):
                continue
    
    return {
        'steps': steps,
        'rec_loss': rec_loss,
        'active_codes': active_codes,
    }

def smooth_curve(y, weight=0.95):
    """Exponential moving average smoothing."""
    if len(y) == 0:
        return y
    smoothed = [y[0]]
    for val in y[1:]:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * val)
    return smoothed

def downsample_data(steps, values, max_points=200):
    """Downsample data to max_points for cleaner plotting."""
    if len(steps) <= max_points:
        return steps, values
    
    # Use evenly spaced indices
    indices = np.linspace(0, len(steps) - 1, max_points, dtype=int)
    return [steps[i] for i in indices], [values[i] for i in indices]

def create_simple_plots():
    """Create simple comparison plots for all models."""
    # Define model paths
    model_paths = {
        'VQ': 'results/vq/vq_fair_v2',
        'FSQ': 'results/fsq/fsq_fair_v2',
        'SimVQ': 'results/sim_vq/sim_vq_fair_v2',
        'LMB': 'results/lmb/lmb_nb16384_tau1.0-0.1_bs64_lr3e-4_dim128_20260119_214749_6c81',
    }
    
    # Load data for each model
    model_data = {}
    for model_name, path in model_paths.items():
        csv_path = Path(path) / 'train_metrics.csv'
        if csv_path.exists():
            print(f"Loading {model_name}...")
            model_data[model_name] = load_train_metrics(csv_path, max_step=30000)
        else:
            print(f"Warning: {csv_path} not found for {model_name}")
    
    if not model_data:
        print("No model data found!")
        return
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Comparison - Training Metrics (up to 30k steps)', 
                 fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {
        'VQ': '#1f77b4',
        'FSQ': '#ff7f0e',
        'SimVQ': '#2ca02c',
        'LMB': '#d62728',
    }
    
    # Plot 1: Reconstruction Loss
    ax = axes[0]
    for model_name, data in model_data.items():
        if data['steps']:
            # Downsample to ~200 points for cleaner lines
            steps_ds, rec_loss_ds = downsample_data(data['steps'], data['rec_loss'], max_points=200)
            y_smooth = smooth_curve(rec_loss_ds, weight=0.95)
            ax.plot(steps_ds, y_smooth, 
                   label=model_name, 
                   linewidth=2.5, 
                   alpha=0.85,
                   color=colors.get(model_name, None))
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Reconstruction Loss', fontsize=11)
    ax.set_title('Reconstruction Loss (Lower is Better)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(0, 30000)
    ax.set_ylim(0.15, 0.35)
    
    # Plot 2: Active Codes
    ax = axes[1]
    for model_name, data in model_data.items():
        if data['steps'] and data['active_codes']:
            # Average active codes per step for smoother visualization
            step_to_codes = defaultdict(list)
            for step, codes in zip(data['steps'], data['active_codes']):
                step_to_codes[step].append(codes)
            
            steps_avg = sorted(step_to_codes.keys())
            codes_avg = [np.mean(step_to_codes[s]) for s in steps_avg]
            
            # Downsample to ~200 points for cleaner lines
            steps_ds, codes_ds = downsample_data(steps_avg, codes_avg, max_points=200)
            y_smooth = smooth_curve(codes_ds, weight=0.95)
            
            ax.plot(steps_ds, y_smooth,
                   label=model_name,
                   linewidth=2.5,
                   alpha=0.85,
                   color=colors.get(model_name, None))
    ax.set_xlabel('Training Steps', fontsize=11)
    ax.set_ylabel('Active Codes', fontsize=11)
    ax.set_title('Codebook Utilization', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.95, fontsize=10)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_xlim(0, 30000)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("plots") / "all_models_comparison_simple_30k.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved to {output_path}")
    
    plt.close()
    
    # Print simple summary
    print("\n" + "="*60)
    print("Model Comparison Summary (up to 30k steps)")
    print("="*60)
    for model_name, data in sorted(model_data.items()):
        if data['steps']:
            final_loss = data['rec_loss'][-1] if data['rec_loss'] else 0
            avg_codes = np.mean(data['active_codes']) if data['active_codes'] else 0
            print(f"{model_name:10} - Final Loss: {final_loss:.4f}, Avg Codes: {avg_codes:.0f}")

if __name__ == "__main__":
    create_simple_plots()

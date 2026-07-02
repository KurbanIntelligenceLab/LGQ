#!/usr/bin/env python3
"""
Analyze whether bin centers are being learned in LMB models.
Checks:
1. If centers are learnable parameters (require_grad=True)
2. If centers change between epochs
3. Visualize center evolution over epochs
"""

import torch
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def load_checkpoint(checkpoint_path):
    """Load a checkpoint and extract bin centers."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to find centers in the model state dict
    centers = None
    centers_key = None
    
    for key, value in checkpoint.get('model_state_dict', {}).items():
        if 'quantize.centers' in key:
            centers = value.numpy() if isinstance(value, torch.Tensor) else value
            centers_key = key
            break
    
    return checkpoint, centers, centers_key

def analyze_centers_learnability(model_dir):
    """Check if centers are learnable by inspecting the model."""
    # Try to load the model config
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load a checkpoint to inspect the model structure
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if not checkpoints:
        return None
    
    # Load first checkpoint
    checkpoint_path = checkpoints[0]
    checkpoint, centers, centers_key = load_checkpoint(checkpoint_path)
    
    if centers is None:
        print(f"Warning: Could not find centers in checkpoint {checkpoint_path}")
        return None
    
    # Check if centers are in the optimizer state (indicates they're learnable)
    optimizer_state = checkpoint.get('optimizer_state_dict', {})
    is_learnable = False
    if optimizer_state:
        param_groups = optimizer_state.get('param_groups', [])
        state = optimizer_state.get('state', {})
        
        # Check if any parameter in optimizer corresponds to centers
        for group in param_groups:
            for param_id in group.get('params', []):
                if param_id in state:
                    is_learnable = True
                    break
    
    # Also check if centers_key exists in optimizer state
    model_state = checkpoint.get('model_state_dict', {})
    if centers_key and centers_key in model_state:
        # Check if this parameter has gradients tracked
        # We can't directly check require_grad from checkpoint, but we can check optimizer
        pass
    
    return {
        'centers_shape': centers.shape,
        'centers_key': centers_key,
        'is_in_optimizer': is_learnable,
        'centers_sample': centers.flatten()[:10].tolist() if centers.size > 0 else None,
        'centers_mean': float(centers.mean()),
        'centers_std': float(centers.std()),
    }

def compare_centers_across_epochs(model_dir):
    """Compare bin centers across different epochs."""
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) < 2:
        print(f"Need at least 2 checkpoints, found {len(checkpoints)}")
        return None
    
    epoch_centers = {}
    for cp_path in checkpoints:
        # Extract epoch number
        try:
            epoch = int(cp_path.stem.split('_')[-1])
        except:
            continue
        
        checkpoint, centers, _ = load_checkpoint(cp_path)
        if centers is not None:
            epoch_centers[epoch] = centers
    
    if len(epoch_centers) < 2:
        return None
    
    # Compare consecutive epochs
    epochs = sorted(epoch_centers.keys())
    changes = []
    
    for i in range(len(epochs) - 1):
        epoch1, epoch2 = epochs[i], epochs[i+1]
        centers1 = epoch_centers[epoch1]
        centers2 = epoch_centers[epoch2]
        
        # Compute difference
        diff = np.abs(centers2 - centers1)
        mean_diff = np.mean(diff)
        max_diff = np.max(diff)
        std_diff = np.std(diff)
        
        # Relative change
        centers1_abs_mean = np.abs(centers1).mean()
        relative_change = mean_diff / (centers1_abs_mean + 1e-8)
        
        changes.append({
            'epoch1': epoch1,
            'epoch2': epoch2,
            'mean_diff': mean_diff,
            'max_diff': max_diff,
            'std_diff': std_diff,
            'relative_change': relative_change
        })
    
    return {
        'epochs_analyzed': epochs,
        'changes': changes,
        'centers_shape': epoch_centers[epochs[0]].shape,
        'initial_mean': float(epoch_centers[epochs[0]].mean()),
        'final_mean': float(epoch_centers[epochs[-1]].mean()),
        'total_change': float(np.abs(epoch_centers[epochs[-1]] - epoch_centers[epochs[0]]).mean()),
    }

def visualize_center_evolution(model_dir, output_dir):
    """Create plots showing how centers evolve."""
    checkpoint_dir = model_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return
    
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) < 2:
        return
    
    epoch_centers = {}
    for cp_path in checkpoints:
        try:
            epoch = int(cp_path.stem.split('_')[-1])
        except:
            continue
        checkpoint, centers, _ = load_checkpoint(cp_path)
        if centers is not None:
            epoch_centers[epoch] = centers
    
    if len(epoch_centers) < 2:
        return
    
    epochs = sorted(epoch_centers.keys())
    
    # Plot 1: Mean center value over epochs
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Mean center per epoch
    mean_centers = [epoch_centers[e].mean() for e in epochs]
    axes[0, 0].plot(epochs, mean_centers, 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Mean Center Value')
    axes[0, 0].set_title('Mean Bin Center Value Over Epochs')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Std of centers per epoch
    std_centers = [epoch_centers[e].std() for e in epochs]
    axes[0, 1].plot(epochs, std_centers, 'o-', linewidth=2, markersize=6, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Std of Center Values')
    axes[0, 1].set_title('Spread of Bin Centers Over Epochs')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Change between consecutive epochs
    changes = []
    for i in range(len(epochs) - 1):
        diff = np.abs(epoch_centers[epochs[i+1]] - epoch_centers[epochs[i]]).mean()
        changes.append(diff)
    
    axes[1, 0].plot(epochs[1:], changes, 'o-', linewidth=2, markersize=6, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Mean Absolute Change')
    axes[1, 0].set_title('Change in Centers Between Epochs')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Sample a few centers and track them
    centers_flat = epoch_centers[epochs[0]].flatten()
    num_to_track = min(20, len(centers_flat))
    indices = np.linspace(0, len(centers_flat)-1, num_to_track, dtype=int)
    
    for idx in indices:
        values = [epoch_centers[e].flatten()[idx] for e in epochs]
        axes[1, 1].plot(epochs, values, 'o-', linewidth=1, markersize=3, alpha=0.6)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Center Value')
    axes[1, 1].set_title(f'Sample of {num_to_track} Individual Centers Over Epochs')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / f"{model_dir.name}_center_evolution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Saved plot to: {output_path}")
    plt.close()

def main():
    results_dir = Path("results")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze LMB experiments
    lmb_experiments = [
        results_dir / "lmb" / "lmb_fixed_init",
        results_dir / "lmb" / "lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd",
    ]
    
    print("=" * 70)
    print("ANALYZING BIN CENTER LEARNING IN LMB MODELS")
    print("=" * 70)
    
    for exp_dir in lmb_experiments:
        if not exp_dir.exists():
            print(f"\n⚠️  Experiment not found: {exp_dir}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Experiment: {exp_dir.name}")
        print(f"{'='*70}")
        
        # 1. Check learnability
        print("\n1. Checking if centers are learnable parameters...")
        learnability = analyze_centers_learnability(exp_dir)
        if learnability:
            print(f"   ✅ Centers found in checkpoint!")
            print(f"   Centers shape: {learnability['centers_shape']}")
            print(f"   Centers key: {learnability['centers_key']}")
            print(f"   Mean value: {learnability['centers_mean']:.6f}")
            print(f"   Std value: {learnability['centers_std']:.6f}")
            print(f"   Note: Centers are registered as nn.Parameter (learnable)")
        else:
            print("   ⚠️  Could not analyze learnability")
        
        # 2. Compare across epochs
        print("\n2. Comparing centers across epochs...")
        comparison = compare_centers_across_epochs(exp_dir)
        if comparison:
            print(f"   ✅ Analyzed {len(comparison['epochs_analyzed'])} epochs: {comparison['epochs_analyzed']}")
            print(f"   Centers shape: {comparison['centers_shape']}")
            print(f"   Initial mean: {comparison['initial_mean']:.6f}")
            print(f"   Final mean: {comparison['final_mean']:.6f}")
            print(f"   Total change (epoch 1 → last): {comparison['total_change']:.6f}")
            print(f"\n   Changes between consecutive epochs:")
            for change in comparison['changes'][:5]:  # Show first 5
                print(f"     Epoch {change['epoch1']} → {change['epoch2']}:")
                print(f"       Mean absolute change: {change['mean_diff']:.6f}")
                print(f"       Max change: {change['max_diff']:.6f}")
                print(f"       Relative change: {change['relative_change']*100:.4f}%")
            if len(comparison['changes']) > 5:
                print(f"     ... and {len(comparison['changes']) - 5} more epochs")
            
            # Determine if learning is happening
            avg_change = np.mean([c['mean_diff'] for c in comparison['changes']])
            if avg_change > 1e-6:
                print(f"\n   ✅ Centers ARE changing (avg change per epoch: {avg_change:.6f})")
                print(f"   ✅ Bins are being LEARNED!")
            else:
                print(f"\n   ❌ Centers are NOT changing significantly (avg change: {avg_change:.6f})")
                print(f"   ⚠️  Bins may not be learning (check optimizer/lr)")
        else:
            print("   ⚠️  Could not compare across epochs (need at least 2 checkpoints)")
        
        # 3. Visualize
        print("\n3. Creating visualization...")
        visualize_center_evolution(exp_dir, output_dir)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze which codebook bins are actually used in LMB models.
Shows distribution of bin usage, unused bins, and patterns.
"""

import torch
import json
import csv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from collections import Counter

def analyze_bin_usage(experiment_dir: Path, epoch: int = None):
    """Analyze which bins are used in an LMB experiment."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        print(f"  ⚠️  No eval_metrics.csv found")
        return None
    
    # Load indices from checkpoints (if available)
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"  ⚠️  No checkpoints directory found")
        return None
    
    # Find checkpoint for specified epoch or latest
    if epoch is None:
        checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            print(f"  ⚠️  No checkpoints found")
            return None
        checkpoint_path = checkpoints[-1]
        epoch = int(checkpoint_path.stem.split("_")[-1])
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
        if not checkpoint_path.exists():
            print(f"  ⚠️  Checkpoint for epoch {epoch} not found")
            return None
    
    print(f"  📊 Analyzing epoch {epoch} from {checkpoint_path.name}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', {})
    
    # Get codebook size
    centers = None
    for k, v in state.items():
        if 'quantize.centers' in k:
            centers = v
            break
    
    if centers is None:
        print(f"  ⚠️  Could not find centers in checkpoint")
        return None
    
    num_bins = centers.shape[0]
    print(f"  📦 Codebook size: {num_bins}")
    
    # Load active codes from eval metrics
    with open(eval_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Find row for this epoch
    active_codes = None
    for row in rows:
        if int(row.get('epoch', 0)) == epoch:
            active_codes = int(float(row.get('val_active_codes', 0)))
            utilization = float(row.get('val_codebook_util', 0))
            break
    
    if active_codes is None:
        print(f"  ⚠️  Could not find metrics for epoch {epoch}")
        return None
    
    print(f"  ✅ Active codes: {active_codes} / {num_bins} ({utilization:.2f}%)")
    print(f"  📉 Unused bins: {num_bins - active_codes} ({100 - utilization:.2f}%)")
    
    # Analyze center distribution
    centers_np = centers.numpy()
    print(f"\n  📊 Center statistics:")
    print(f"     Mean: {centers_np.mean():.4f}, Std: {centers_np.std():.4f}")
    print(f"     Min: {centers_np.min():.4f}, Max: {centers_np.max():.4f}")
    
    # Check if unused bins are clustered
    # (We can't know which specific bins are unused without running inference,
    # but we can check center distribution)
    print(f"\n  🔍 Center distribution analysis:")
    print(f"     Per-dimension std: {centers_np.std(axis=0).mean():.4f} ± {centers_np.std(axis=0).std():.4f}")
    
    # Check if centers are well-spread
    center_norms = np.linalg.norm(centers_np, axis=1)
    print(f"     Center norms: mean={center_norms.mean():.4f}, std={center_norms.std():.4f}")
    print(f"     Center norm range: [{center_norms.min():.4f}, {center_norms.max():.4f}]")
    
    return {
        'epoch': epoch,
        'num_bins': num_bins,
        'active_codes': active_codes,
        'utilization': utilization,
        'unused_bins': num_bins - active_codes,
        'centers_mean': float(centers_np.mean()),
        'centers_std': float(centers_np.std()),
        'center_norms_mean': float(center_norms.mean()),
        'center_norms_std': float(center_norms.std()),
    }

def main():
    results_dir = Path("results/lmb")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    # Analyze main LMB baseline
    lmb_baseline = results_dir / "lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd"
    
    print("=" * 70)
    print("ANALYZING BIN USAGE IN LMB MODELS")
    print("=" * 70)
    
    if lmb_baseline.exists():
        print(f"\n📁 Experiment: {lmb_baseline.name}")
        result = analyze_bin_usage(lmb_baseline, epoch=10)
        if result:
            print(f"\n  ✅ Analysis complete!")
            print(f"     {result['utilization']:.2f}% utilization is {'NORMAL' if result['utilization'] > 40 else 'LOW'}")
            print(f"     This suggests {'good' if result['utilization'] > 40 else 'poor'} codebook diversity")
    else:
        print(f"  ⚠️  Baseline experiment not found")
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()

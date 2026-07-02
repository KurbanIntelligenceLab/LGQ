#!/usr/bin/env python3
"""
Plot bin center learning dynamics for LMB models.

Visualizes:
1. Bin center positions over epochs
2. Distribution of bin centers
3. Per-channel bin evolution
4. Codebook visualization (for flattened mode)
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, List
import argparse
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

def load_centers_from_checkpoint(checkpoint_path: Path) -> Optional[torch.Tensor]:
    """Load bin centers from a checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model_state = checkpoint.get('model_state_dict', {})
        
        for key, value in model_state.items():
            if 'quantize.centers' in key:
                return value
        return None
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def plot_tsne_initial_vs_final(exp_dir: Path, output_dir: Path, max_points: int = 2000, perplexity: float = 30.0, initial_checkpoint_path: Optional[Path] = None, use_nn_matching: bool = False):
    """Create a t-SNE plot showing initial vs final bin centers with connecting lines.
    
    Args:
        exp_dir: Experiment directory
        output_dir: Output directory for plots
        max_points: Maximum number of points for t-SNE
        perplexity: t-SNE perplexity parameter
        initial_checkpoint_path: Optional path to initial checkpoint (if not found in exp_dir)
        use_nn_matching: If True, match bins by nearest neighbor (for different experiments).
                        If False, match by index (for same experiment - default LMB behavior)
    """
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"  ⚠️  No checkpoints directory found")
        return
    
    # Load config to understand mode
    config_path = exp_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    flatten_channels = config.get('flatten_channels', False)
    embedding_dim = config.get('embedding_dim', 128)
    
    # Load initial and final centers
    initial_centers = None
    final_centers = None
    initial_epoch = None
    final_epoch = None
    
    # If initial_checkpoint_path is provided, use it for initial state
    if initial_checkpoint_path is not None and initial_checkpoint_path.exists():
        try:
            centers = load_centers_from_checkpoint(initial_checkpoint_path)
            if centers is not None:
                checkpoint = torch.load(initial_checkpoint_path, map_location='cpu')
                epoch = checkpoint.get('epoch', 1)
                initial_centers = centers.cpu().numpy()
                initial_epoch = epoch
                print(f"  ✓ Using initial checkpoint from: {initial_checkpoint_path} (epoch {epoch})")
        except Exception as e:
            print(f"  ⚠️  Error loading initial checkpoint from {initial_checkpoint_path}: {e}")
    
    # Find all checkpoints - check both numbered checkpoints and best/latest
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    
    # Also check best_model.pt and latest_model.pt if they exist
    best_model_path = checkpoint_dir / "best_model.pt"
    latest_model_path = checkpoint_dir / "latest_model.pt"
    
    # Load all available checkpoints with their epochs
    checkpoint_data = []
    
    # Load numbered checkpoints
    for cp_path in checkpoints:
        try:
            epoch_str = cp_path.stem.split('_')[-1]
            epoch = int(epoch_str)
            centers = load_centers_from_checkpoint(cp_path)
            if centers is not None:
                checkpoint_data.append((epoch, centers.cpu().numpy(), cp_path))
        except Exception as e:
            continue
    
    # Load best_model.pt if it exists
    if best_model_path.exists():
        try:
            checkpoint = torch.load(best_model_path, map_location='cpu')
            epoch = checkpoint.get('epoch', None)
            if epoch is not None:
                centers = load_centers_from_checkpoint(best_model_path)
                if centers is not None:
                    checkpoint_data.append((epoch, centers.cpu().numpy(), best_model_path))
        except Exception as e:
            pass
    
    # Load latest_model.pt if it exists
    if latest_model_path.exists():
        try:
            checkpoint = torch.load(latest_model_path, map_location='cpu')
            epoch = checkpoint.get('epoch', None)
            if epoch is not None:
                centers = load_centers_from_checkpoint(latest_model_path)
                if centers is not None:
                    checkpoint_data.append((epoch, centers.cpu().numpy(), latest_model_path))
        except Exception as e:
            pass
    
    # Remove duplicates (keep first occurrence of each epoch)
    seen_epochs = set()
    unique_checkpoints = []
    for epoch, centers, path in sorted(checkpoint_data, key=lambda x: x[0]):
        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            unique_checkpoints.append((epoch, centers, path))
    
    # If we already have initial_centers from external checkpoint, we just need final
    if initial_centers is not None:
        if len(unique_checkpoints) < 1:
            print(f"  ⚠️  Need at least 1 checkpoint for final state, found {len(unique_checkpoints)}")
            return
        # Use the latest checkpoint as final
        unique_checkpoints.sort(key=lambda x: x[0])
        final_epoch, final_centers, _ = unique_checkpoints[-1]
    else:
        if len(unique_checkpoints) < 2:
            print(f"  ⚠️  Need at least 2 checkpoints, found {len(unique_checkpoints)}")
            if len(unique_checkpoints) > 0:
                print(f"  Available epochs: {[e for e, _, _ in unique_checkpoints]}")
            return
        # Sort by epoch and get initial and final
        unique_checkpoints.sort(key=lambda x: x[0])
        initial_epoch, initial_centers, _ = unique_checkpoints[0]
        final_epoch, final_centers, _ = unique_checkpoints[-1]
    
    if initial_centers is None or final_centers is None:
        print(f"  ⚠️  Could not load initial and final centers")
        return
    
    # Reshape centers based on mode
    if flatten_channels:
        # Flattened mode: centers shape is [K, C] where K is num_bins, C is embedding_dim
        # Each row is a codebook vector
        initial_flat = initial_centers  # [K, C]
        final_flat = final_centers  # [K, C]
    else:
        # Per-channel mode: centers shape is [C, K] where C is embedding_dim, K is num_bins
        # Need to transpose to get [K, C] where each row is a codebook vector
        initial_flat = initial_centers.T  # [K, C]
        final_flat = final_centers.T  # [K, C]
    
    # Ensure we have the same number of centers
    num_centers = min(len(initial_flat), len(final_flat))
    initial_flat = initial_flat[:num_centers]
    final_flat = final_flat[:num_centers]
    
    # Match bins: by index (default, for same experiment) or by nearest neighbor (for different experiments)
    if use_nn_matching:
        # Match bins by nearest neighbor in original space
        # Use this when comparing checkpoints from different experiments where bins may not correspond by index
        print(f"  🔗 Matching {num_centers} bins by nearest neighbor (different experiments)...")
        nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn.fit(final_flat)
        distances, matched_indices = nn.kneighbors(initial_flat)
        matched_indices = matched_indices.flatten()
        
        # Reorder final_flat to match initial_flat
        final_flat_matched = final_flat[matched_indices]
    else:
        # Match bins by index (default LMB behavior - bins keep their indices during training)
        # Use this when comparing epochs from the same experiment
        print(f"  🔗 Matching {num_centers} bins by index (same experiment)...")
        final_flat_matched = final_flat
    
    # Sample points if too many (sample from initial, keep corresponding final)
    if num_centers > max_points:
        sample_idx = np.linspace(0, num_centers - 1, max_points, dtype=int)
        initial_flat = initial_flat[sample_idx]
        final_flat_matched = final_flat_matched[sample_idx]
        num_centers = max_points
        print(f"  📊 Sampled to {num_centers} points for visualization")
    
    print(f"  📊 Computing t-SNE for {num_centers} centers (dim={initial_flat.shape[1]})...")
    
    # Combine initial and final for consistent t-SNE embedding
    # This ensures they're in the same coordinate space
    all_centers = np.vstack([initial_flat, final_flat_matched])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=min(perplexity, num_centers - 1), 
                random_state=42, max_iter=1000, verbose=0)
    all_embedded = tsne.fit_transform(all_centers)
    
    # Split back into initial and final
    initial_embedded = all_embedded[:num_centers]
    final_embedded = all_embedded[num_centers:]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw connecting lines (sample a subset to avoid clutter)
    num_lines = min(500, num_centers)
    line_indices = np.linspace(0, num_centers - 1, num_lines, dtype=int)
    for idx in line_indices:
        ax.plot([initial_embedded[idx, 0], final_embedded[idx, 0]],
                [initial_embedded[idx, 1], final_embedded[idx, 1]],
                'g-', alpha=0.1, linewidth=0.5, zorder=1)
    
    # Plot initial centers as blue circles
    ax.scatter(initial_embedded[:, 0], initial_embedded[:, 1],
              c='blue', marker='o', s=20, alpha=0.4, 
              label=f'Initial (Epoch {initial_epoch})', zorder=2)
    
    # Plot final centers as red 'x' marks
    ax.scatter(final_embedded[:, 0], final_embedded[:, 1],
              c='red', marker='x', s=20, alpha=0.4,
              label=f'Final (Epoch {final_epoch})', zorder=3)
    
    ax.set_xlabel('t-SNE Dimension 0', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 1', fontsize=12)
    ax.set_title('2D Projection: Initial vs Final (t-SNE)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    output_path = output_dir / f"{exp_dir.name}_tsne_initial_vs_final.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved t-SNE plot to: {output_path}")
    plt.close()

def plot_bin_centers_evolution(exp_dir: Path, output_dir: Path, max_epochs: int = 20):
    """Plot how bin centers evolve over training epochs."""
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"  ⚠️  No checkpoints directory found")
        return
    
    # Load config to understand mode
    config_path = exp_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    
    flatten_channels = config.get('flatten_channels', False)
    num_bins = config.get('num_bins', 16384)
    embedding_dim = config.get('embedding_dim', 128)
    
    # Find all checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) < 2:
        print(f"  ⚠️  Need at least 2 checkpoints, found {len(checkpoints)}")
        return
    
    # Load centers from each checkpoint
    epoch_centers = {}
    epochs = []
    for cp_path in checkpoints[:max_epochs]:
        try:
            epoch = int(cp_path.stem.split('_')[-1])
            centers = load_centers_from_checkpoint(cp_path)
            if centers is not None:
                epoch_centers[epoch] = centers.cpu().numpy()
                epochs.append(epoch)
        except Exception as e:
            print(f"  ⚠️  Error processing {cp_path}: {e}")
            continue
    
    if len(epoch_centers) < 2:
        print(f"  ⚠️  Could not load enough checkpoints")
        return
    
    epochs = sorted(epochs)
    
    # Create comprehensive plots
    if flatten_channels:
        # Flattened mode: centers shape is [K, C]
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Mean center value over epochs
        ax1 = plt.subplot(3, 2, 1)
        mean_centers = [epoch_centers[e].mean() for e in epochs]
        ax1.plot(epochs, mean_centers, 'o-', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Center Value')
        ax1.set_title('Mean Bin Center Value Over Epochs')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Std of centers
        ax2 = plt.subplot(3, 2, 2)
        std_centers = [epoch_centers[e].std() for e in epochs]
        ax2.plot(epochs, std_centers, 'o-', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Std of Center Values')
        ax2.set_title('Spread of Bin Centers Over Epochs')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Change between epochs
        ax3 = plt.subplot(3, 2, 3)
        changes = []
        for i in range(len(epochs) - 1):
            diff = np.abs(epoch_centers[epochs[i+1]] - epoch_centers[epochs[i]]).mean()
            changes.append(diff)
        ax3.plot(epochs[1:], changes, 'o-', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Mean Absolute Change')
        ax3.set_title('Change in Centers Between Epochs')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of centers at different epochs
        ax4 = plt.subplot(3, 2, 4)
        selected_epochs = [epochs[0], epochs[len(epochs)//2], epochs[-1]]
        for epoch in selected_epochs:
            centers_flat = epoch_centers[epoch].flatten()
            ax4.hist(centers_flat, bins=50, alpha=0.5, label=f'Epoch {epoch}')
        ax4.set_xlabel('Center Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Bin Centers')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Sample individual centers over epochs
        ax5 = plt.subplot(3, 2, 5)
        centers_flat = epoch_centers[epochs[0]].flatten()
        num_to_track = min(50, len(centers_flat))
        indices = np.linspace(0, len(centers_flat)-1, num_to_track, dtype=int)
        for idx in indices:
            values = [epoch_centers[e].flatten()[idx] for e in epochs]
            ax5.plot(epochs, values, 'o-', linewidth=0.5, markersize=2, alpha=0.3)
        ax5.set_xlabel('Epoch')
        ax5.set_ylabel('Center Value')
        ax5.set_title(f'Sample of {num_to_track} Individual Centers')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: 2D projection of codebook (first 2 dimensions)
        ax6 = plt.subplot(3, 2, 6)
        for epoch in selected_epochs:
            centers = epoch_centers[epoch]
            # Sample a subset for visualization
            if len(centers) > 1000:
                sample_idx = np.linspace(0, len(centers)-1, 1000, dtype=int)
                centers = centers[sample_idx]
            ax6.scatter(centers[:, 0], centers[:, 1], alpha=0.3, s=1, label=f'Epoch {epoch}')
        ax6.set_xlabel('Dimension 0')
        ax6.set_ylabel('Dimension 1')
        ax6.set_title('2D Projection of Codebook (First 2 Dims)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
    else:
        # Per-channel mode: centers shape is [C, K]
        fig = plt.figure(figsize=(16, 12))
        
        # Plot 1: Mean center value over epochs
        ax1 = plt.subplot(3, 2, 1)
        mean_centers = [epoch_centers[e].mean() for e in epochs]
        ax1.plot(epochs, mean_centers, 'o-', linewidth=2, markersize=6, color='blue')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Center Value')
        ax1.set_title('Mean Bin Center Value Over Epochs')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Std of centers
        ax2 = plt.subplot(3, 2, 2)
        std_centers = [epoch_centers[e].std() for e in epochs]
        ax2.plot(epochs, std_centers, 'o-', linewidth=2, markersize=6, color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Std of Center Values')
        ax2.set_title('Spread of Bin Centers Over Epochs')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Change between epochs
        ax3 = plt.subplot(3, 2, 3)
        changes = []
        for i in range(len(epochs) - 1):
            diff = np.abs(epoch_centers[epochs[i+1]] - epoch_centers[epochs[i]]).mean()
            changes.append(diff)
        ax3.plot(epochs[1:], changes, 'o-', linewidth=2, markersize=6, color='green')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Mean Absolute Change')
        ax3.set_title('Change in Centers Between Epochs')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribution of centers
        ax4 = plt.subplot(3, 2, 4)
        selected_epochs = [epochs[0], epochs[len(epochs)//2], epochs[-1]]
        for epoch in selected_epochs:
            centers_flat = epoch_centers[epoch].flatten()
            ax4.hist(centers_flat, bins=50, alpha=0.5, label=f'Epoch {epoch}')
        ax4.set_xlabel('Center Value')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Distribution of Bin Centers')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Per-channel bin positions (sample a few channels)
        ax5 = plt.subplot(3, 2, 5)
        num_channels_to_plot = min(5, embedding_dim)
        for c in range(num_channels_to_plot):
            for epoch in selected_epochs:
                centers = epoch_centers[epoch][c, :]  # [K]
                sorted_centers = np.sort(centers)
                ax5.plot(sorted_centers, alpha=0.5, label=f'Ch {c}, Epoch {epoch}' if epoch == selected_epochs[0] else '')
        ax5.set_xlabel('Bin Index (sorted)')
        ax5.set_ylabel('Bin Center Value')
        ax5.set_title(f'Bin Positions for First {num_channels_to_plot} Channels')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Heatmap of centers for a sample channel
        ax6 = plt.subplot(3, 2, 6)
        channel_idx = 0
        center_matrix = np.array([epoch_centers[e][channel_idx, :] for e in epochs])
        im = ax6.imshow(center_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        ax6.set_xlabel('Bin Index')
        ax6.set_ylabel('Epoch')
        ax6.set_title(f'Bin Center Evolution for Channel {channel_idx}')
        plt.colorbar(im, ax=ax6)
    
    plt.suptitle(f'Bin Center Learning: {exp_dir.name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / f"{exp_dir.name}_bin_learning.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ Saved plot to: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot bin center learning dynamics")
    parser.add_argument("--exp-dir", type=str, help="Path to experiment directory")
    parser.add_argument("--output-dir", type=str, default="plots/bin_learning", help="Output directory for plots")
    parser.add_argument("--max-epochs", type=int, default=20, help="Maximum epochs to plot")
    parser.add_argument("--all-lmb", action="store_true", help="Plot all LMB experiments")
    parser.add_argument("--tsne-only", action="store_true", help="Only create t-SNE initial vs final plot")
    parser.add_argument("--max-points", type=int, default=2000, help="Maximum points for t-SNE (default: 2000)")
    parser.add_argument("--perplexity", type=float, default=30.0, help="t-SNE perplexity (default: 30.0)")
    parser.add_argument("--initial-checkpoint", type=str, default=None, help="Path to initial checkpoint (if not found in exp-dir)")
    parser.add_argument("--use-nn-matching", action="store_true", help="Use nearest neighbor matching instead of index-based (for different experiments)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.tsne_only:
        # Only create t-SNE plot
        if args.exp_dir:
            exp_dir = Path(args.exp_dir)
            if not exp_dir.exists():
                print(f"Error: Experiment directory not found: {exp_dir}")
                return
            print(f"Creating t-SNE plot for: {exp_dir.name}")
            initial_cp = Path(args.initial_checkpoint) if args.initial_checkpoint else None
            # Only use NN matching if explicitly requested
            use_nn = args.use_nn_matching
            plot_tsne_initial_vs_final(exp_dir, output_dir, args.max_points, args.perplexity, initial_cp, use_nn)
        elif args.all_lmb:
            results_dir = Path("results/lmb")
            if results_dir.exists():
                exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
                print(f"Found {len(exp_dirs)} LMB experiments")
            for exp_dir in exp_dirs:
                print(f"\nCreating t-SNE plot for: {exp_dir.name}")
                initial_cp = Path(args.initial_checkpoint) if args.initial_checkpoint else None
                use_nn = args.use_nn_matching
                plot_tsne_initial_vs_final(exp_dir, output_dir, args.max_points, args.perplexity, initial_cp, use_nn)
        else:
            print("Please specify --exp-dir or use --all-lmb")
            parser.print_help()
    elif args.all_lmb:
        # Find all LMB experiments
        results_dir = Path("results/lmb")
        if results_dir.exists():
            exp_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
            print(f"Found {len(exp_dirs)} LMB experiments")
            for exp_dir in exp_dirs:
                print(f"\nPlotting: {exp_dir.name}")
                plot_bin_centers_evolution(exp_dir, output_dir, args.max_epochs)
                # Also create t-SNE plot
                initial_cp = Path(args.initial_checkpoint) if args.initial_checkpoint else None
                use_nn = args.use_nn_matching
                plot_tsne_initial_vs_final(exp_dir, output_dir, args.max_points, args.perplexity, initial_cp, use_nn)
    elif args.exp_dir:
        exp_dir = Path(args.exp_dir)
        if not exp_dir.exists():
            print(f"Error: Experiment directory not found: {exp_dir}")
            return
        print(f"Plotting: {exp_dir.name}")
        plot_bin_centers_evolution(exp_dir, output_dir, args.max_epochs)
        # Also create t-SNE plot
        initial_cp = Path(args.initial_checkpoint) if args.initial_checkpoint else None
        use_nn = args.use_nn_matching
        plot_tsne_initial_vs_final(exp_dir, output_dir, args.max_points, args.perplexity, initial_cp, use_nn)
    else:
        print("Please specify --exp-dir or use --all-lmb")
        parser.print_help()

if __name__ == "__main__":
    main()

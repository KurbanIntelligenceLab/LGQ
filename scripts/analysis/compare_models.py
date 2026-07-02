#!/usr/bin/env python3
"""Compare model metrics across different quantization methods."""

import csv
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Results directories
RESULTS_ROOT = Path("/workspace/vector-quantize/results")


def find_latest_run(model_dir: Path, dim: int = 128) -> Path | None:
    """Find the latest run directory for a model with the given dimension."""
    runs = list(model_dir.glob(f"*_dim{dim}_*"))
    if not runs:
        return None
    # Sort by modification time (latest first)
    runs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    # Find one with eval_metrics.csv
    for run in runs:
        if (run / "eval_metrics.csv").exists():
            return run
    return None


def load_eval_metrics(run_dir: Path) -> list[dict] | None:
    """Load evaluation metrics from a run directory as list of dicts."""
    eval_file = run_dir / "eval_metrics.csv"
    if not eval_file.exists():
        return None
    with open(eval_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # Convert numeric values
            converted = {}
            for k, v in row.items():
                try:
                    if '.' in v:
                        converted[k] = float(v)
                    else:
                        converted[k] = int(v)
                except (ValueError, TypeError):
                    converted[k] = v
            rows.append(converted)
        return rows


def get_model_name(run_dir: Path) -> str:
    """Extract a clean model name from the run directory."""
    name = run_dir.parent.name  # e.g., 'fsq', 'vq', 'lmb', 'sim_vq'
    return name.upper().replace('_', '-')


def create_comparison_table(dim: int = 128):
    """Create a comparison table of all models at the same epochs."""
    models = {}
    
    for model_type in ['fsq', 'vq', 'lmb', 'sim_vq']:
        model_dir = RESULTS_ROOT / model_type
        if not model_dir.exists():
            continue
        
        run_dir = find_latest_run(model_dir, dim=dim)
        if run_dir is None:
            print(f"No run found for {model_type} with dim={dim}")
            continue
        
        rows = load_eval_metrics(run_dir)
        if rows is None:
            continue
        
        model_name = get_model_name(run_dir)
        models[model_name] = {
            'dir': run_dir,
            'rows': rows
        }
        print(f"Loaded {model_name}: {run_dir.name} ({len(rows)} epochs)")
    
    if not models:
        print("No models found!")
        return
    
    # Find common epochs
    all_epochs = [set(r['epoch'] for r in m['rows']) for m in models.values()]
    common_epochs = set.intersection(*all_epochs) if all_epochs else set()
    
    if not common_epochs:
        print("\nNo common epochs found. Showing available epochs per model:")
        for name, data in models.items():
            print(f"  {name}: epochs {sorted([r['epoch'] for r in data['rows']])}")
        print("\nUsing all available epochs for each model:")
        common_epochs = None
    
    # Metrics to compare
    metrics = ['val_rfid', 'val_psnr', 'val_ssim', 'val_lpips', 'val_rec_loss']
    metric_labels = ['rFID ↓', 'PSNR ↑', 'SSIM ↑', 'LPIPS ↓', 'Rec Loss ↓']
    
    # Build comparison data
    comparison_data = []
    
    for name, data in sorted(models.items()):
        rows = data['rows']
        if common_epochs:
            epochs_to_use = sorted(common_epochs)
        else:
            epochs_to_use = sorted([r['epoch'] for r in rows])
        
        for epoch in epochs_to_use:
            row = next((r for r in rows if r['epoch'] == epoch), None)
            if row is None:
                continue
            
            entry = {'Model': name, 'Epoch': int(epoch)}
            for metric, label in zip(metrics, metric_labels):
                if metric in row:
                    val = row[metric]
                    if 'fid' in metric.lower():
                        entry[label] = f"{val:.2f}"
                    else:
                        entry[label] = f"{val:.4f}"
                else:
                    entry[label] = "N/A"
            
            # Handle codebook utilization
            # LMB: effective codebook = bins × channels = 128 × 128 = 16,384
            # Others: val_active_codes / total codebook
            codebook_util = row.get('val_codebook_util', 0)
            if codebook_util > 0:
                entry['Util %'] = f"{codebook_util:.1f}%"
            else:
                # Calculate from active codes if not available
                active = int(row.get('val_active_codes', 0))
                if name == 'LMB':
                    # For LMB, active codes was incorrectly stored as num_bins in old runs
                    # Effective codebook = 128 × 128 = 16,384
                    eff_size = dim * dim  # 128 * 128
                    entry['Util %'] = "100.0%"  # LMB uses all (channel, bin) pairs
                else:
                    util = 100.0 * active / 16384 if active > 0 else 0
                    entry['Util %'] = f"{util:.1f}%"
            
            comparison_data.append(entry)
    
    # Print table
    print(f"\n{'='*120}")
    print(f"Model Comparison Table (dim={dim})")
    print(f"{'='*120}")
    
    if comparison_data:
        # Get column headers
        headers = list(comparison_data[0].keys())
        col_widths = {h: max(len(h), max(len(str(row.get(h, ''))) for row in comparison_data)) for h in headers}
        
        # Print header
        header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows
        for row in comparison_data:
            row_line = " | ".join(str(row.get(h, '')).ljust(col_widths[h]) for h in headers)
            print(row_line)
    
    # Save to CSV
    output_file = RESULTS_ROOT.parent / "plots" / f"model_comparison_dim{dim}.csv"
    output_file.parent.mkdir(exist_ok=True)
    
    if comparison_data:
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparison_data[0].keys()))
            writer.writeheader()
            writer.writerows(comparison_data)
        print(f"\nSaved to: {output_file}")
    
    return comparison_data


def plot_fid_over_epochs(dim: int = 128):
    """Plot FID over epochs for all models."""
    plt.figure(figsize=(12, 8))
    
    colors = {
        'FSQ': '#1f77b4',
        'VQ': '#ff7f0e', 
        'LMB': '#2ca02c',
        'SIM-VQ': '#d62728'
    }
    
    markers = {
        'FSQ': 'o',
        'VQ': 's',
        'LMB': '^',
        'SIM-VQ': 'D'
    }
    
    has_data = False
    
    for model_type in ['fsq', 'vq', 'lmb', 'sim_vq']:
        model_dir = RESULTS_ROOT / model_type
        if not model_dir.exists():
            continue
        
        run_dir = find_latest_run(model_dir, dim=dim)
        if run_dir is None:
            continue
        
        rows = load_eval_metrics(run_dir)
        if rows is None:
            continue
        
        # Check if val_rfid exists
        if not any('val_rfid' in r for r in rows):
            continue
        
        model_name = get_model_name(run_dir)
        
        epochs = [r['epoch'] for r in rows if 'val_rfid' in r]
        fids = [r['val_rfid'] for r in rows if 'val_rfid' in r]
        
        if epochs and fids:
            has_data = True
            plt.plot(
                epochs, 
                fids,
                label=model_name,
                color=colors.get(model_name, 'gray'),
                marker=markers.get(model_name, 'o'),
                markersize=8,
                linewidth=2,
                alpha=0.8
            )
    
    if not has_data:
        print("No FID data found for any model!")
        plt.close()
        return
    
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('rFID (lower is better)', fontsize=14)
    plt.title(f'rFID Comparison Across Models (dim={dim})', fontsize=16)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_file = RESULTS_ROOT.parent / "plots" / f"fid_comparison_dim{dim}.png"
    output_file.parent.mkdir(exist_ok=True)
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved FID plot to: {output_file}")
    
    plt.close()


def main():
    dim = 128
    if len(sys.argv) > 1:
        dim = int(sys.argv[1])
    
    print(f"Comparing models with dim={dim}\n")
    
    # Create comparison table
    create_comparison_table(dim=dim)
    
    # Create FID plot
    print("\nGenerating FID plot...")
    plot_fid_over_epochs(dim=dim)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Plot quantization error (latent→code distance) vs codebook utilization across epochs.

X-axis: Utilization (%)
Y-axis: Latent→code distance (mean L2 distance)

Shows how well each quantization method balances codebook usage with quantization fidelity
as they learn over multiple epochs. Lower error at higher utilization is better.

Usage:
    python scripts/plot_error_vs_utilization.py --main --epochs
    python scripts/plot_error_vs_utilization.py --experiment-dirs <dirs...> --epochs
"""

from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import functions from visualize_latent_vs_codes.py
from scripts.visualize_latent_vs_codes import (
    FlatImageDataset,
    get_transform,
    load_model,
    collect_z_and_active,
    get_active_code_vectors,
    compute_metrics,
    MAIN_CHECKPOINTS,
)

# Model colors and markers
MODEL_STYLE = {
    "fsq": {"color": "#1f77b4", "marker": "o", "label": "FSQ"},
    "vq": {"color": "#ff7f0e", "marker": "s", "label": "VQ"},
    "lfq": {"color": "#2ca02c", "marker": "^", "label": "LFQ"},
    "sim_vq": {"color": "#d62728", "marker": "D", "label": "SimVQ"},
    "lmb": {"color": "#9467bd", "marker": "X", "label": "LMB"},
}


def find_epoch_checkpoints(experiment_dir: Path) -> List[Tuple[int, Path]]:
    """Find all epoch checkpoints in an experiment directory."""
    checkpoint_dir = experiment_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    
    checkpoints = []
    # Find numbered checkpoints
    for cp_path in sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt")):
        match = re.search(r"checkpoint_epoch_(\d+)\.pt", cp_path.name)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, cp_path))
    
    # Also check best_model.pt and latest_model.pt
    best_model = checkpoint_dir / "best_model.pt"
    latest_model = checkpoint_dir / "latest_model.pt"
    
    if best_model.exists():
        try:
            ckpt = torch.load(best_model, map_location="cpu")
            epoch = ckpt.get("epoch", -1)
            if epoch > 0:
                checkpoints.append((epoch, best_model))
        except:
            pass
    
    if latest_model.exists():
        try:
            ckpt = torch.load(latest_model, map_location="cpu")
            epoch = ckpt.get("epoch", -1)
            if epoch > 0:
                checkpoints.append((epoch, latest_model))
        except:
            pass
    
    # Sort by epoch; dedupe by epoch (keep first path for each epoch)
    seen_epochs = set()
    unique = []
    for epoch, path in sorted(checkpoints, key=lambda x: x[0]):
        if epoch not in seen_epochs:
            seen_epochs.add(epoch)
            unique.append((epoch, path))
    return unique


def plot_error_vs_utilization_epochs(
    model_data: Dict[str, List[Tuple[int, float, float]]],  # model_type -> [(epoch, util, error), ...]
    output_path: Path,
    codebook_size: int = 16384,
    exclude_lfq: bool = True,
    log_scale: bool = False,
) -> None:
    """Plot quantization error vs utilization trajectories across epochs."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Colormap for epochs (light to dark)
    cmap = plt.cm.viridis
    
    # Collect all epochs to normalize colormap
    all_epochs = []
    for epochs_data in model_data.values():
        all_epochs.extend([e for e, _, _ in epochs_data])
    if all_epochs:
        epoch_min, epoch_max = min(all_epochs), max(all_epochs)
    else:
        epoch_min, epoch_max = 1, 10
    
    # Plot trajectories
    for model_type, epochs_data in model_data.items():
        if not epochs_data:
            continue
        
        # Skip LFQ if requested (it has very high error)
        if exclude_lfq and model_type == "lfq":
            continue
        
        style = MODEL_STYLE.get(model_type, {"color": "#808080", "marker": "o", "label": model_type.upper()})
        
        # Sort by epoch
        epochs_data = sorted(epochs_data, key=lambda x: x[0])
        epochs = [e for e, _, _ in epochs_data]
        utils = [u for _, u, _ in epochs_data]
        errors = [err for _, _, err in epochs_data]
        
        # Plot line connecting epochs
        ax.plot(
            utils,
            errors,
            color=style["color"],
            linewidth=2.5,
            alpha=0.6,
            label=style["label"],
            zorder=3,
        )
        
        # Plot points with color indicating epoch
        for epoch, util, error in epochs_data:
            # Normalize epoch to [0, 1] for colormap
            epoch_norm = (epoch - epoch_min) / max(1, epoch_max - epoch_min)
            point_color = cmap(epoch_norm)
            
            ax.scatter(
                util,
                error,
                color=point_color,
                marker=style["marker"],
                s=150,
                alpha=0.8,
                edgecolors=style["color"],
                linewidths=2,
                zorder=5,
            )
        
        # Annotate first and last epochs
        if len(epochs_data) > 0:
            first_epoch, first_util, first_error = epochs_data[0]
            last_epoch, last_util, last_error = epochs_data[-1]
            
            # First epoch
            ax.annotate(
                f"E{first_epoch}",
                (first_util, first_error),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=style["color"],
                alpha=0.7,
            )
            
            # Last epoch (if different)
            if last_epoch != first_epoch:
                ax.annotate(
                    f"E{last_epoch}",
                    (last_util, last_error),
                    xytext=(5, -15),
                    textcoords="offset points",
                    fontsize=8,
                    color=style["color"],
                    fontweight="bold",
                )
    
    ax.set_xlabel("Codebook Utilization (%)", fontsize=14, fontweight="bold")
    ylabel = "Latent→Code Distance (L2)"
    if log_scale:
        ylabel += " (log scale)"
        ax.set_yscale("log")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(
        "Quantization Error vs Codebook Utilization (Across Epochs)\n"
        "Lower error at higher utilization is better. Points colored by epoch (light→dark).",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits
    all_utils = []
    all_errors = []
    for epochs_data in model_data.values():
        for _, util, error in epochs_data:
            if not (exclude_lfq and error > 100):  # Skip LFQ outliers if excluding
                all_utils.append(util)
                all_errors.append(error)
    
    if all_utils and all_errors:
        util_min, util_max = min(all_utils), max(all_utils)
        error_min, error_max = min(all_errors), max(all_errors)
        
        util_padding = (util_max - util_min) * 0.1
        error_padding = (error_max - error_min) * 0.1
        
        ax.set_xlim(max(0, util_min - util_padding), min(100, util_max + util_padding))
        if not log_scale:
            ax.set_ylim(max(0, error_min - error_padding), error_max + error_padding)
    
    # Add legend
    ax.legend(loc="best", fontsize=11, framealpha=0.9, edgecolor="black", fancybox=True)
    
    # Add colorbar for epochs
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=epoch_min, vmax=epoch_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Epoch", pad=0.02)
    cbar.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot to: {output_path}")


def plot_error_vs_utilization(
    model_data: Dict[str, Dict],
    output_path: Path,
    codebook_size: int = 16384,
) -> None:
    """Plot quantization error vs utilization for all models (single epoch version)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Collect data points
    for model_type, data in model_data.items():
        utilization = 100.0 * data["n_active"] / codebook_size
        error = data["metrics"]["latent_to_code_mean"]
        
        style = MODEL_STYLE.get(model_type, {"color": "gray", "marker": "o", "label": model_type.upper()})
        
        ax.scatter(
            utilization,
            error,
            color=style["color"],
            marker=style["marker"],
            s=200,
            alpha=0.7,
            edgecolors="black",
            linewidths=1.5,
            label=style["label"],
            zorder=5,
        )
        
        # Add text annotation with model name
        ax.annotate(
            style["label"],
            (utilization, error),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            color=style["color"],
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=style["color"], linewidth=1.5),
        )
    
    ax.set_xlabel("Codebook Utilization (%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Latent→Code Distance (L2)", fontsize=14, fontweight="bold")
    ax.set_title(
        "Quantization Error vs Codebook Utilization\n"
        "Lower error at higher utilization is better",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Set axis limits with some padding
    utilizations = [100.0 * data["n_active"] / codebook_size for data in model_data.values()]
    errors = [data["metrics"]["latent_to_code_mean"] for data in model_data.values()]
    
    util_min, util_max = min(utilizations), max(utilizations)
    error_min, error_max = min(errors), max(errors)
    
    util_padding = (util_max - util_min) * 0.1
    error_padding = (error_max - error_min) * 0.1
    
    ax.set_xlim(max(0, util_min - util_padding), min(100, util_max + util_padding))
    ax.set_ylim(max(0, error_min - error_padding), error_max + error_padding)
    
    # Add legend
    ax.legend(loc="best", fontsize=11, framealpha=0.9, edgecolor="black", fancybox=True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved plot to: {output_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Plot quantization error vs utilization for all models"
    )
    ap.add_argument(
        "--main",
        action="store_true",
        help="Use main experiment directories for all models (16K)",
    )
    ap.add_argument(
        "--experiment-dirs",
        nargs="+",
        type=str,
        default=[],
        help="Experiment directories (one per model, order: fsq, vq, lfq, sim_vq, lmb)",
    )
    ap.add_argument(
        "--epochs",
        action="store_true",
        help="Plot across multiple epochs (finds all checkpoint_epoch_*.pt files)",
    )
    ap.add_argument("--data-root", type=str, default="data/imagenet")
    ap.add_argument("--num-images", type=int, default=200)
    ap.add_argument("--subsample-latents", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--output-dir", type=str, default="results/plots")
    ap.add_argument(
        "--codebook-size",
        type=int,
        default=16384,
        help="Codebook size for utilization calculation",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        type=str,
        default=None,
        help="Restrict to these models (e.g. sim_vq lmb). Default: all in main checkpoints.",
    )
    ap.add_argument(
        "--include-lfq",
        action="store_true",
        help="Include LFQ in plot (default: exclude, as it has very high error due to different code space)",
    )
    ap.add_argument(
        "--log-scale",
        action="store_true",
        help="Use log scale for y-axis",
    )
    ap.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum number of epochs to process per model",
    )
    args = ap.parse_args()

    # Determine experiment directories
    if args.main:
        # Extract experiment directories from MAIN_CHECKPOINTS
        experiment_dirs = {}
        for model_type, ckpt_path in MAIN_CHECKPOINTS.items():
            ckpt_path = project_root / ckpt_path
            if ckpt_path.exists():
                experiment_dirs[model_type] = ckpt_path.parent.parent
        if args.models:
            experiment_dirs = {k: v for k, v in experiment_dirs.items() if k in args.models}
        print("Using main experiment directories:")
        for k, v in experiment_dirs.items():
            print(f"  {k.upper()}: {v}")
    elif args.experiment_dirs:
        if len(args.experiment_dirs) != 5:
            print(f"Warning: Expected 5 experiment dirs, got {len(args.experiment_dirs)}")
        model_order = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
        experiment_dirs = {}
        for model_type, exp_dir in zip(model_order, args.experiment_dirs):
            exp_path = Path(exp_dir)
            if not exp_path.is_absolute():
                exp_path = project_root / exp_path
            experiment_dirs[model_type] = exp_path
    else:
        raise SystemExit("Provide --experiment-dirs or use --main")

    device = "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser()
    test_path = data_root / "test"
    if not test_path.exists():
        test_path = data_root / "val"
    transform = get_transform(128)
    dataset = FlatImageDataset(
        str(test_path), transform=transform, max_images=args.num_images * 2
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    if args.epochs:
        # Multi-epoch mode
        all_model_data: Dict[str, List[Tuple[int, float, float]]] = {}
        
        for model_type, exp_dir in experiment_dirs.items():
            if not exp_dir.exists():
                print(f"⚠️  Skipping {model_type}: experiment dir not found: {exp_dir}")
                continue
            
            print("\n" + "=" * 60)
            print(f"{model_type.upper()} - Finding checkpoints...")
            print("=" * 60)
            
            checkpoints = find_epoch_checkpoints(exp_dir)
            # Fallback: if no numbered checkpoints, use latest/best as single "epoch"
            if not checkpoints:
                checkpoint_dir = exp_dir / "checkpoints"
                for name, key in [("latest_model.pt", "epoch"), ("best_model.pt", "epoch")]:
                    cp = checkpoint_dir / name
                    if cp.exists():
                        try:
                            ckpt = torch.load(cp, map_location="cpu")
                            epoch = ckpt.get("epoch", 1)
                            checkpoints = [(epoch, cp)]
                            print(f"  Using {name} as epoch {epoch}")
                            break
                        except Exception:
                            pass
            if not checkpoints:
                print(f"  ⚠️  No checkpoints found in {exp_dir}")
                continue
            
            if args.max_epochs:
                checkpoints = checkpoints[:args.max_epochs]
            
            print(f"  Found {len(checkpoints)} checkpoints")
            
            epochs_data = []
            for epoch, ckpt_path in tqdm(checkpoints, desc=f"Processing {model_type}"):
                try:
                    model, config, detected_type = load_model(str(ckpt_path), device)
                    if detected_type != model_type:
                        print(f"  ⚠️  Warning: Expected {model_type}, got {detected_type}")
                    
                    z, n_active, active_data, _ = collect_z_and_active(
                        model,
                        loader,
                        device,
                        model_type,
                        num_images=args.num_images,
                        subsample_latents=args.subsample_latents,
                    )
                    e = get_active_code_vectors(model, model_type, active_data, device)
                    metrics = compute_metrics(z, e)
                    
                    utilization = 100.0 * n_active / args.codebook_size
                    error = metrics["latent_to_code_mean"]
                    
                    epochs_data.append((epoch, utilization, error))
                    
                    print(f"  Epoch {epoch}: util={utilization:.2f}%, error={error:.4f}")
                    
                    del model
                except Exception as e:
                    print(f"  ❌ Error processing epoch {epoch}: {e}")
                    continue
            
            if epochs_data:
                all_model_data[model_type] = epochs_data
        
        if len(all_model_data) == 0:
            raise RuntimeError("No models were successfully processed")
        
        # Create plot
        print("\n" + "=" * 60)
        print("Creating error vs utilization plot (multi-epoch)...")
        print("=" * 60)
        
        output_path = out_dir / "error_vs_utilization_epochs.png"
        plot_error_vs_utilization_epochs(
            all_model_data,
            output_path,
            args.codebook_size,
            exclude_lfq=not args.include_lfq,
            log_scale=args.log_scale,
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for model_type, epochs_data in sorted(all_model_data.items()):
            if epochs_data:
                first_epoch, first_util, first_error = epochs_data[0]
                last_epoch, last_util, last_error = epochs_data[-1]
                style = MODEL_STYLE.get(model_type, {}).get("label", model_type.upper())
                print(f"\n{style}:")
                print(f"  Epoch {first_epoch}: util={first_util:.2f}%, error={first_error:.4f}")
                print(f"  Epoch {last_epoch}: util={last_util:.2f}%, error={last_error:.4f}")
                print(f"  Change: util={last_util-first_util:+.2f}%, error={last_error-first_error:+.4f}")
    
    else:
        # Single epoch mode (original behavior)
        all_model_data: Dict[str, Dict] = {}
        
        for model_type, exp_dir in experiment_dirs.items():
            if not exp_dir.exists():
                print(f"⚠️  Skipping {model_type}: experiment dir not found: {exp_dir}")
                continue
            
            # Find latest checkpoint
            checkpoint_dir = exp_dir / "checkpoints"
            latest_model = checkpoint_dir / "latest_model.pt"
            if not latest_model.exists():
                latest_model = checkpoint_dir / "best_model.pt"
            
            if not latest_model.exists():
                print(f"⚠️  Skipping {model_type}: no checkpoint found in {checkpoint_dir}")
                continue
            
            print("\n" + "=" * 60)
            print(f"{model_type.upper()} (16K)")
            print("=" * 60)
            
            try:
                model, config, detected_type = load_model(str(latest_model), device)
                if detected_type != model_type:
                    print(f"  ⚠️  Warning: Expected {model_type}, got {detected_type}")
                
                z, n_active, active_data, _ = collect_z_and_active(
                    model,
                    loader,
                    device,
                    model_type,
                    num_images=args.num_images,
                    subsample_latents=args.subsample_latents,
                )
                e = get_active_code_vectors(model, model_type, active_data, device)
                metrics = compute_metrics(z, e)
                
                utilization = 100.0 * n_active / args.codebook_size
                error = metrics["latent_to_code_mean"]
                
                print(f"  Active codes: {n_active}")
                print(f"  Utilization: {utilization:.2f}%")
                print(f"  Latent→code distance: {error:.4f}")
                
                all_model_data[model_type] = {
                    "z": z,
                    "e": e,
                    "n_active": n_active,
                    "metrics": metrics,
                    "name": style.get("label", model_type.upper()) if (style := MODEL_STYLE.get(model_type)) else model_type.upper(),
                }
                
                del model
            except Exception as e:
                print(f"  ❌ Error processing {model_type}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(all_model_data) == 0:
            raise RuntimeError("No models were successfully processed")
        
        # Create plot
        print("\n" + "=" * 60)
        print("Creating error vs utilization plot...")
        print("=" * 60)
        
        output_path = out_dir / "error_vs_utilization.png"
        plot_error_vs_utilization(all_model_data, output_path, args.codebook_size)
        
        # Print summary table
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        print(f"{'Model':<15} {'Utilization':<15} {'Error':<15}")
        print("-" * 60)
        for model_type, data in sorted(all_model_data.items()):
            util = 100.0 * data["n_active"] / args.codebook_size
            error = data["metrics"]["latent_to_code_mean"]
            name = data["name"]
            print(f"{name:<15} {util:>10.2f}%    {error:>12.4f}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create elbow (WCSS) and Silhouette vs k plots with each model placed on them.

Draws vertical lines at each model's active code count and labels (model name + k)
so you can compare where VQ, LMB, FSQ, SimVQ, LFQ sit relative to the latent
space structure.

Outputs:
  - plots/elbow_silhouette_with_models.png  (one figure, two subplots)

Usage:
  python scripts/plot_elbow_silhouette_with_models.py
  python scripts/plot_elbow_silhouette_with_models.py --num-images 200 -o plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from scripts.visualize_latent_vs_codes import (
    MAIN_CHECKPOINTS,
    discover_lmb_fair_checkpoint,
    load_model,
    get_transform,
    FlatImageDataset,
    collect_z_and_active,
    get_active_code_vectors,
    compute_metrics,
)
from torch.utils.data import DataLoader


def run_kmeans_sweep(z_np: np.ndarray, k_values: list[int], max_samples: int = 50000, random_state: int = 42) -> dict:
    """Run K-means for each k; return inertia (WCSS) and silhouette per k."""
    if z_np.shape[0] > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(z_np.shape[0], max_samples, replace=False)
        z_sub = z_np[idx]
    else:
        z_sub = z_np
    results = {}
    for k in tqdm(k_values, desc="K-means sweep"):
        if k >= z_sub.shape[0]:
            results[k] = {"inertia": np.nan, "silhouette": np.nan}
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = km.fit_predict(z_sub)
        results[k] = {
            "inertia": float(km.inertia_),
            "silhouette": float(silhouette_score(z_sub, labels, sample_size=min(5000, z_sub.shape[0]), random_state=random_state)),
        }
    return results


def main():
    ap = argparse.ArgumentParser(description="Plot elbow and silhouette vs k with models placed on them")
    ap.add_argument("--num-images", type=int, default=300)
    ap.add_argument("--subsample-latents", type=int, default=20000)
    ap.add_argument("--k-values", type=str, default="1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000",
                    help="Comma-separated k values for sweep")
    ap.add_argument("-o", "--output-dir", type=Path, default=Path("plots"))
    ap.add_argument("--output-name", type=str, default="elbow_silhouette_with_models.png")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.output_name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    fair = discover_lmb_fair_checkpoint()
    if fair is not None:
        checkpoints["lmb_fair"] = fair
    model_order = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
    checkpoints = {k: v for k, v in checkpoints.items() if k in model_order or k == "lmb_fair"}

    data_root = project_root / "data"
    test_path = data_root / "test"
    if not test_path.exists():
        test_path = data_root / "val"
    if not test_path.exists():
        test_path = data_root / "imagenet" / "test"
    if not test_path.exists():
        test_path = data_root / "imagenet" / "val"
    if not test_path.exists():
        test_path = data_root

    transform = get_transform(128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    colors = {"fsq": "#ff7f0e", "vq": "#1f77b4", "lfq": "#9467bd", "sim_vq": "#2ca02c", "lmb": "#d62728"}
    names = {"fsq": "FSQ 16K", "vq": "VQ 16K", "lfq": "LFQ 16K", "sim_vq": "SimVQ 16K", "lmb": "LMB 16K", "lmb_fair": "LMB 16K"}

    results_per_model = {}
    for model_type in model_order:
        if model_type not in checkpoints or not checkpoints[model_type].exists():
            continue
        ckpt_path = checkpoints[model_type]
        print(f"Loading {model_type}...")
        try:
            model, config, detected_type = load_model(str(ckpt_path), device)
            effective_type = detected_type if model_type == "lmb_fair" else model_type
            z, n_active, active_data, idx_sub = collect_z_and_active(
                model, loader, device, effective_type,
                num_images=args.num_images,
                subsample_latents=args.subsample_latents,
            )
            if effective_type == "fsq":
                z = model.quantize.bound(z.to(device)).cpu()
            z_np = z.numpy()
            results_per_model[model_type] = {"name": names[model_type], "n_active": n_active, "z": z_np}
            del model
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if len(results_per_model) == 0:
        print("No models loaded. Exiting.")
        return

    first_key = next(iter(results_per_model))
    z_sweep = results_per_model[first_key]["z"]
    print("Running K-means sweep...")
    sweep = run_kmeans_sweep(z_sweep, k_values)
    ks = sorted(sweep.keys())
    inertias = [sweep[k]["inertia"] for k in ks]
    sils = [sweep[k]["silhouette"] for k in ks]

    active_counts = {m: results_per_model[m]["n_active"] for m in results_per_model}
    model_order_plot = [m for m in model_order if m in results_per_model]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # --- WCSS vs k ---
    ax1.plot(ks, inertias, "k-o", markersize=5, linewidth=1.5, label="K-means on encoder z")
    ymax_wcss = max(inertias) if inertias else 1
    for i, m in enumerate(model_order_plot):
        k = active_counts[m]
        c = colors.get(m, "gray")
        ax1.axvline(k, color=c, linestyle="--", alpha=0.9, linewidth=1.2)
        ax1.text(k, ymax_wcss * (1.02 + 0.04 * (i % 2)), f"{results_per_model[m]['name']}\n(k={k})",
                 fontsize=8, color=c, ha="center", va="bottom", rotation=45 if k < 2500 else 0)
    ax1.set_ylabel("Within-cluster variance (WCSS / inertia)")
    ax1.set_title("Elbow: WCSS vs k — vertical lines = each model's active code count")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Silhouette vs k ---
    ax2.plot(ks, sils, "k-o", markersize=5, linewidth=1.5, label="Silhouette on encoder z")
    ymin_sil = min(sils) if sils else 0
    ymax_sil = max(sils) if sils else 1
    for i, m in enumerate(model_order_plot):
        k = active_counts[m]
        c = colors.get(m, "gray")
        ax2.axvline(k, color=c, linestyle="--", alpha=0.9, linewidth=1.2)
        ax2.text(k, ymax_sil + (ymax_sil - ymin_sil) * (0.05 + 0.03 * (i % 2)), f"{results_per_model[m]['name']}\n(k={k})",
                 fontsize=8, color=c, ha="center", va="bottom", rotation=45 if k < 2500 else 0)
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette vs k (higher = better-defined clusters) — vertical lines = each model's active k")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(ymin_sil - 0.02, ymax_sil + 0.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Check "codes concentrate where mass is" in BIN / ASSIGNMENT space.

Idea: For each code (bin), we have:
  - usage = number of z tokens that assign to this code.
  - density_at_code = k-NN density of z evaluated at the code's location (bin center).

If bins that get used more are bins in denser z regions, then usage and density_at_code
should be positively correlated. So we compute Spearman(usage, density_at_code) per model.

Interpretation:
  - High positive correlation => "bins that are used a lot are bins where z has high density"
    => codes (bins) concentrate where encoder mass is, in assignment-space sense.
  - Compare LMB vs VQ vs SimVQ; LMB might show stronger correlation if it adapts geometry.

Outputs: plots/bin_usage_vs_density_metrics.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

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
)
from torch.utils.data import DataLoader


def knn_density_at_points(z_np: np.ndarray, query_np: np.ndarray, k: int = 10) -> np.ndarray:
    """Density at query points: 1/r^d, r = mean dist to k nearest z."""
    d = z_np.shape[1]
    n_neighbors = min(k, z_np.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(z_np)
    dists, _ = nn.kneighbors(query_np)
    r = np.mean(dists, axis=1)
    r = np.maximum(r, 1e-10)
    return 1.0 / (r ** d)


def main():
    ap = argparse.ArgumentParser(description="Check usage vs density in bin/assignment space")
    ap.add_argument("--num-images", type=int, default=150)
    ap.add_argument("--subsample-latents", type=int, default=8000)
    ap.add_argument("--k-nn", type=int, default=10)
    ap.add_argument("--output-dir", type=str, default="plots")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    fair = discover_lmb_fair_checkpoint()
    if fair is not None:
        checkpoints["lmb_fair"] = fair
    model_order = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
    checkpoints = {k: v for k, v in checkpoints.items() if k in model_order or k == "lmb_fair"}

    data_root = project_root / "data"
    for sub in ("test", "val", "imagenet/test", "imagenet/val", ""):
        test_path = data_root / sub if sub else data_root
        if test_path.exists():
            break
    else:
        test_path = data_root

    transform = get_transform(128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    results = {}

    for model_type, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"Skipping {model_type}: checkpoint not found")
            continue
        print(f"Processing {model_type}...")
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
            code_device = "cpu"
            if code_device != device:
                model = model.cpu()
            e = get_active_code_vectors(model, effective_type, active_data, code_device)

            z_np = z.numpy()
            e_np = e.numpy()

            # Per-code usage: for each active code, how many z assign to it?
            if effective_type == "lmb":
                config = getattr(model, "config", {})
                lmb_flattened = getattr(model.quantize, "flatten_channels", False) if hasattr(model, "quantize") else config.get("flatten_channels", False)
                if lmb_flattened:
                    flat_idx = idx_sub.reshape(-1).numpy()
                    # active_data is list of int (code indices)
                    usage = np.array([np.sum(flat_idx == aid) for aid in active_data], dtype=np.float64)
                else:
                    # Per-channel: active_data is list of tuples; idx_sub is (N, C)
                    idx_arr = idx_sub.numpy()
                    usage = np.array([np.sum(np.all(idx_arr == aid, axis=1)) for aid in active_data], dtype=np.float64)
            else:
                flat_idx = idx_sub.reshape(-1).numpy()
                # active_data: list of active code indices
                usage = np.array([np.sum(flat_idx == aid) for aid in active_data], dtype=np.float64)

            # Density at each code (bin center)
            density_at_code = knn_density_at_points(z_np, e_np, k=args.k_nn)

            # Correlation: do high-usage bins have high density at their center?
            if usage.size > 2 and np.var(usage) > 0 and np.var(density_at_code) > 0:
                rho, pval = spearmanr(usage, density_at_code)
            else:
                rho, pval = np.nan, np.nan

            name = {"fsq": "FSQ 16K", "vq": "VQ 16K", "lfq": "LFQ 16K", "sim_vq": "SimVQ 16K", "lmb": "LMB 16K", "lmb_fair": "LMB 16K per-ch"}.get(model_type, model_type)
            results[model_type] = {
                "name": name,
                "n_active": n_active,
                "spearman_usage_vs_density": rho,
                "pvalue": pval,
                "mean_usage": float(np.mean(usage)),
                "mean_density_at_codes": float(np.mean(density_at_code)),
            }
            del model
        except Exception as e:
            print(f"Error {model_type}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No models processed.")
        return

    # Report
    lines = [
        "Bin/assignment space: do high-usage bins sit in high-density z regions?",
        "=" * 70,
        "",
        "For each code (bin): usage = # of z that assign to it; density_at_code = k-NN density of z at bin center.",
        "Spearman(usage, density_at_code): positive => bins that are used more are in denser z regions.",
        "",
        "If LMB's 'codes concentrate where mass is' in BIN space, we expect LMB to have",
        "  high positive Spearman (high-usage bins = high density at bin center).",
        "",
        "-" * 70,
        f"{'Model':<14} {'Spearman(use,dens)':>18} {'p-value':>10} {'n_active':>10}",
        "-" * 70,
    ]
    for m in model_order:
        if m not in results:
            continue
        r = results[m]
        rho = r["spearman_usage_vs_density"]
        pval = r["pvalue"]
        rho_str = f"{rho:.4f}" if not np.isnan(rho) else "N/A"
        pstr = f"{pval:.2e}" if not np.isnan(pval) and pval < 1 else (f"{pval:.4f}" if not np.isnan(pval) else "N/A")
        lines.append(f"{r['name']:<14} {rho_str:>18} {pstr:>10} {r['n_active']:>10}")
    lines.append("-" * 70)

    # Interpretation: among models with valid correlation, which has highest?
    valid = [(m, results[m]["spearman_usage_vs_density"]) for m in model_order if m in results and not np.isnan(results[m]["spearman_usage_vs_density"])]
    if valid:
        best = max(valid, key=lambda x: x[1])
        lines.append("")
        lines.append(f"Highest Spearman (usage vs density at code): {results[best[0]]['name']} (rho = {best[1]:.4f})")
        if best[0] == "lmb":
            lines.append("=> Supports: LMB's high-usage bins are in high-density z regions (concentration in bin space).")
        else:
            lines.append("=> LMB is not the highest; concentration-in-bin-space interpretation is mixed.")
    lines.append("")
    lines.append("=" * 70)

    out_txt = out_dir / "bin_usage_vs_density_metrics.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()

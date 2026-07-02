#!/usr/bin/env python3
"""
Check whether each model's codes concentrate where encoder mass is (geometry adapts).

For each model we have z (encoder outputs) and e (active code vectors) in the same space.
- Density at z: k-NN density estimate for each z point (density ∝ 1/r^d where r = mean dist to k nearest z).
- Density at codes: for each code e_j, density(e_j) = 1 / (mean dist from e_j to k nearest z)^d.
- Ratio = mean(density at codes) / mean(density at z). If > 1, codes tend to sit in higher-density regions.
- Mean code-to-latent: mean over codes of distance to nearest z (smaller = codes closer to mass).

If "LMB's codes concentrate where encoder mass is", we expect LMB to have:
  - Ratio > 1 and possibly highest among models (codes in denser regions).
  - Low mean code-to-latent (codes close to z).

Outputs: plots/code_density_alignment_metrics.txt and optional bar plot.

With --ablations: run only LMB regularization ablations (reg_none, reg_bins_only,
reg_peak_only, reg_strong, reg_weak) and write plots/code_density_alignment_lmb_ablations.txt
to show controlled utilization / geometry as a function of (lambda_peak, lambda_bins).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
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
)
from torch.utils.data import DataLoader


def knn_density_at_points(z_np: np.ndarray, query_np: np.ndarray, k: int = 10, query_is_z: bool = False) -> np.ndarray:
    """
    For each query point, compute k-NN density: density ∝ 1 / r^d where r = mean distance to k nearest z.
    If query_is_z, query points are z themselves: use k+1 neighbors and drop first (self) so r = mean dist to k other z.
    Returns density per query point.
    """
    d = z_np.shape[1]
    n_neighbors = min(k + 1, z_np.shape[0]) if query_is_z else min(k, z_np.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(z_np)
    dists, _ = nn.kneighbors(query_np)  # (n_query, n_neighbors)
    if query_is_z and dists.shape[1] > 1:
        r = np.mean(dists[:, 1:], axis=1)  # exclude self (distance 0)
    else:
        r = np.mean(dists, axis=1)
    r = np.maximum(r, 1e-10)
    density = 1.0 / (r ** d)
    return density


def code_to_latent_distances(z_np: np.ndarray, e_np: np.ndarray) -> np.ndarray:
    """For each code, distance to nearest z. Shape (n_codes,)."""
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(z_np)
    dists, _ = nn.kneighbors(e_np)
    return dists.flatten()


# LMB regularization ablation runs: run_name -> (display_name, checkpoint subpath preferred)
LMB_ABLATION_RUNS = [
    ("lmb_ablation_reg_none", "None (0, 0)"),
    ("lmb_ablation_reg_weak", "Weak (0.002, 0.002)"),
    ("lmb_ablation_reg_strong", "Strong (0.01, 0.01)"),
    ("lmb_ablation_reg_bins_only", "Bins only (0, 0.01)"),
    ("lmb_ablation_reg_peak_only", "Peak only (0.01, 0)"),
]


def _get_ablation_checkpoints(results_lmb: Path) -> dict[str, Path]:
    """Resolve checkpoint path for each LMB ablation run (latest or epoch-2)."""
    out = {}
    for run_name, _ in LMB_ABLATION_RUNS:
        run_dir = results_lmb / run_name
        if not run_dir.is_dir():
            continue
        for ckpt_name in ("latest_model.pt", "checkpoint_epoch_002.pt"):
            ckpt = run_dir / "checkpoints" / ckpt_name
            if ckpt.exists():
                out[run_name] = ckpt
                break
    return out


def main():
    ap = argparse.ArgumentParser(description="Check if codes concentrate where encoder mass is")
    ap.add_argument("--num-images", type=int, default=150)
    ap.add_argument("--subsample-latents", type=int, default=8000)
    ap.add_argument("--k-nn", type=int, default=10, help="k for k-NN density")
    ap.add_argument("--output-dir", type=str, default="plots")
    ap.add_argument("--ablations", action="store_true",
                    help="Run LMB regularization ablations only (controlled utilization / geometry)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_ablations = args.ablations
    if use_ablations:
        results_lmb = project_root / "results" / "lmb"
        checkpoints = _get_ablation_checkpoints(results_lmb)
        model_order = [r[0] for r in LMB_ABLATION_RUNS if r[0] in checkpoints]
        ablation_display = {r[0]: r[1] for r in LMB_ABLATION_RUNS}
    else:
        checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
        fair = discover_lmb_fair_checkpoint()
        if fair is not None:
            checkpoints["lmb_fair"] = fair
        model_order = ["fsq", "vq", "lfq", "sim_vq", "lmb"]
        checkpoints = {k: v for k, v in checkpoints.items() if k in model_order or k == "lmb_fair"}
        ablation_display = {}

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

    results = {}
    k_nn = args.k_nn

    for model_type, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"Skipping {model_type}: checkpoint not found")
            continue
        print(f"Processing {model_type}...")
        try:
            model, config, detected_type = load_model(str(ckpt_path), device)
            if use_ablations:
                effective_type = "lmb"
            else:
                effective_type = detected_type if model_type == "lmb_fair" else model_type

            z, n_active, active_data, _ = collect_z_and_active(
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

            # Density at z (k-NN: each z point, mean dist to k nearest other z)
            density_z = knn_density_at_points(z_np, z_np, k=k_nn, query_is_z=True)
            mean_density_z = float(np.mean(density_z))

            # Density at codes (each code: mean dist to k nearest z, then density ∝ 1/r^d)
            density_at_codes = knn_density_at_points(z_np, e_np, k=k_nn, query_is_z=False)
            mean_density_codes = float(np.mean(density_at_codes))

            ratio = mean_density_codes / mean_density_z if mean_density_z > 0 else float("nan")

            # Code-to-latent: mean/median distance from each code to nearest z
            code_to_latent = code_to_latent_distances(z_np, e_np)
            mean_code_to_latent = float(np.mean(code_to_latent))
            median_code_to_latent = float(np.median(code_to_latent))

            # Scale-invariant: mean z-to-nearest-z (typical spacing in latent space)
            nn_z = NearestNeighbors(n_neighbors=2, metric="euclidean").fit(z_np)
            dist_z_to_z, _ = nn_z.kneighbors(z_np)
            mean_z_spacing = float(np.mean(dist_z_to_z[:, 1]))
            relative_code_to_latent = mean_code_to_latent / mean_z_spacing if mean_z_spacing > 0 else float("nan")

            if use_ablations:
                name = ablation_display.get(model_type, model_type)
                # Optional: read (lambda_peak, lambda_bins) from run config for table
                run_config = {}
                config_path = ckpt_path.parent.parent / "config.json"
                if config_path.exists():
                    try:
                        with open(config_path) as f:
                            run_config = json.load(f)
                    except Exception:
                        pass
                results[model_type] = {
                    "name": name,
                    "n_active": n_active,
                    "mean_density_z": mean_density_z,
                    "mean_density_codes": mean_density_codes,
                    "density_ratio": ratio,
                    "mean_code_to_latent": mean_code_to_latent,
                    "median_code_to_latent": median_code_to_latent,
                    "mean_z_spacing": mean_z_spacing,
                    "relative_code_to_latent": relative_code_to_latent,
                    "is_discrete": False,
                    "lambda_peak": run_config.get("lambda_peak"),
                    "lambda_bins": run_config.get("lambda_bins"),
                }
            else:
                name = {"fsq": "FSQ 16K", "vq": "VQ 16K", "lfq": "LFQ 16K", "sim_vq": "SimVQ 16K", "lmb": "LMB 16K", "lmb_fair": "LMB 16K per-ch"}.get(model_type, model_type)
                results[model_type] = {
                    "name": name,
                    "n_active": n_active,
                    "mean_density_z": mean_density_z,
                    "mean_density_codes": mean_density_codes,
                    "density_ratio": ratio,
                    "mean_code_to_latent": mean_code_to_latent,
                    "median_code_to_latent": median_code_to_latent,
                    "mean_z_spacing": mean_z_spacing,
                    "relative_code_to_latent": relative_code_to_latent,
                    "is_discrete": effective_type in ("fsq", "lfq"),
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
    def fmt_ratio(x):
        if np.isnan(x) or np.isinf(x):
            return "N/A"
        if x >= 1e6 or (x > 0 and x < 1e-4):
            return f"{x:.2e}"
        return f"{x:.4f}"

    lines = [
        "Code density alignment — do codes concentrate where encoder mass is?",
        "=" * 80,
        "",
        "Density: k-NN estimate (density ∝ 1/r^d, r = mean dist to k nearest z).",
        "density_ratio = mean(density at codes) / mean(density at z).",
        "  ratio > 1  =>  codes tend to sit in higher-density regions than average.",
        "  (FSQ/LFQ: discrete codes → ratio meaningless or huge; focus on continuous methods.)",
        "",
        "mean_code→z = mean over codes of (distance to nearest z).",
        "relative = mean_code→z / mean(z spacing). < 1 => codes closer to z than typical z spacing.",
        "",
        "-" * 80,
        f"{'Model':<14} {'density_ratio':>14} {'mean_code→z':>12} {'relative':>10} {'n_active':>10}",
        "-" * 80,
    ]
    for m in model_order:
        if m not in results:
            continue
        r = results[m]
        ratio_str = fmt_ratio(r["density_ratio"])
        rel_str = f"{r['relative_code_to_latent']:.3f}" if not np.isnan(r["relative_code_to_latent"]) and r["relative_code_to_latent"] < 1e6 else "N/A"
        code_z_str = f"{r['mean_code_to_latent']:.4f}" if r["mean_code_to_latent"] < 1e6 else "N/A"
        lines.append(f"{r['name']:<14} {ratio_str:>14} {code_z_str:>12} {rel_str:>10} {r['n_active']:>10}")
    lines.append("-" * 80)
    lines.append("")
    # Interpretation: among continuous methods only (exclude FSQ, LFQ)
    continuous = [m for m in model_order if m in results and not results[m].get("is_discrete", False) and results[m]["mean_code_to_latent"] < 1e5]
    if continuous:
        best_ratio_m = max((m for m in continuous if not (np.isnan(results[m]["density_ratio"]) or np.isinf(results[m]["density_ratio"]) or results[m]["density_ratio"] > 1e10)), key=lambda m: results[m]["density_ratio"], default=None)
        if best_ratio_m is not None:
            lines.append(f"Among VQ/SimVQ/LMB — highest density_ratio (codes in densest regions): {results[best_ratio_m]['name']} (ratio = {fmt_ratio(results[best_ratio_m]['density_ratio'])})")
        best_close = min(continuous, key=lambda m: results[m]["mean_code_to_latent"])
        lines.append(f"Among VQ/SimVQ/LMB — smallest mean code→z (codes closest to z): {results[best_close]['name']} (mean = {results[best_close]['mean_code_to_latent']:.4f}, relative = {results[best_close]['relative_code_to_latent']:.3f})")
    lines.append("")
    lines.append("Conclusion: If LMB's codes concentrate where encoder mass is, we expect LMB to have")
    lines.append("  density_ratio > 1 and/or relative_code_to_latent < 1 (codes closer than typical z spacing).")
    lines.append("=" * 80)

    if use_ablations:
        # Ablation-specific report: controlled utilization / geometry
        out_txt = out_dir / "code_density_alignment_lmb_ablations.txt"
        ab_lines = [
            "LMB regularization ablations — code density alignment (controlled utilization / geometry)",
            "=" * 80,
            "",
            "Same metrics as main report, but only LMB under different (lambda_peak, lambda_bins).",
            "Shows how regularizers tune geometry: bins → more spread/diversity; peak → more confident assignment.",
            "",
            "density_ratio = mean(density at codes) / mean(density at z).  ratio > 1 => codes in denser regions.",
            "mean_code→z = mean distance from each code to nearest z.  relative = mean_code→z / mean(z spacing).",
            "",
            "-" * 80,
        ]
        # Optional lambda columns if we have them
        has_lambda = any(results.get(m, {}).get("lambda_peak") is not None for m in model_order if m in results)
        if has_lambda:
            ab_lines.append(f"{'Setting':<22} {'λ_peak':>8} {'λ_bins':>8} {'density_ratio':>14} {'mean_code→z':>12} {'relative':>10} {'n_active':>10}")
        else:
            ab_lines.append(f"{'Setting':<22} {'density_ratio':>14} {'mean_code→z':>12} {'relative':>10} {'n_active':>10}")
        ab_lines.append("-" * 80)
        for m in model_order:
            if m not in results:
                continue
            r = results[m]
            ratio_str = fmt_ratio(r["density_ratio"])
            rel_str = f"{r['relative_code_to_latent']:.3f}" if not np.isnan(r["relative_code_to_latent"]) and r["relative_code_to_latent"] < 1e6 else "N/A"
            code_z_str = f"{r['mean_code_to_latent']:.4f}" if r["mean_code_to_latent"] < 1e6 else "N/A"
            name_short = r["name"][:22] if len(r["name"]) > 22 else r["name"]
            if has_lambda:
                lp = r.get("lambda_peak")
                lb = r.get("lambda_bins")
                lp_str = f"{lp:.4f}" if lp is not None else "—"
                lb_str = f"{lb:.4f}" if lb is not None else "—"
                ab_lines.append(f"{name_short:<22} {lp_str:>8} {lb_str:>8} {ratio_str:>14} {code_z_str:>12} {rel_str:>10} {r['n_active']:>10}")
            else:
                ab_lines.append(f"{name_short:<22} {ratio_str:>14} {code_z_str:>12} {rel_str:>10} {r['n_active']:>10}")
        ab_lines.append("-" * 80)
        ab_lines.append("")
        ab_lines.append("Use this table to show: with bins only → more spread (lower density_ratio?); with peak only → different balance.")
        ab_lines.append("=" * 80)
        with open(out_txt, "w") as f:
            f.write("\n".join(ab_lines))
        print("\n" + "\n".join(ab_lines))
        print(f"\nSaved: {out_txt}")
    else:
        out_txt = out_dir / "code_density_alignment_metrics.txt"
        with open(out_txt, "w") as f:
            f.write("\n".join(lines))
        print("\n" + "\n".join(lines))
        print(f"\nSaved: {out_txt}")


if __name__ == "__main__":
    main()

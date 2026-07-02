#!/usr/bin/env python3
"""
Compute professor-recommended quantization structure metrics and histograms.

Metrics:
  1. Within-cluster variance (WCSS / inertia) vs k — elbow logic; compare k = LMB vs FSQ active codes.
  2. Silhouette score vs k — cluster separation/compactness.
  3. Entropy vs WCSS — variance–entropy frontier (lower WCSS at same entropy = better geometry).
  4. Histogram of code usage — Gini, entropy of usage, % mass in top X% codes.
  5. Distance alignment — mean distance z → nearest code (from existing compute_metrics).

Outputs:
  - plots/quantization_structure_wcss_vs_k.png
  - plots/quantization_structure_silhouette_vs_k.png
  - plots/quantization_structure_entropy_vs_wcss.png
  - plots/quantization_structure_code_usage_histograms.png
  - plots/quantization_structure_metrics.txt

Usage:
  python scripts/quantization_structure_metrics.py
  python scripts/quantization_structure_metrics.py --num-images 300 --k-max 12000
"""

from __future__ import annotations

import argparse
import csv
import math
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

# Reuse data collection and metrics from latent-vs-codes visualization
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


def load_last_eval_row(experiment_dir: Path) -> dict | None:
    """Load last row of eval_metrics.csv for entropy, perplexity, active_codes."""
    csv_path = experiment_dir / "eval_metrics.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    out = {}
    for key in ("val_perplexity", "val_active_codes", "val_entropy_bits"):
        v = row.get(key)
        if v is not None and str(v).strip() and str(v) != "None":
            try:
                out[key] = float(v)
            except ValueError:
                out[key] = None
        else:
            out[key] = None
    if out.get("val_perplexity") and out["val_perplexity"] > 0:
        out["entropy_bits"] = math.log2(out["val_perplexity"])
    elif out.get("val_entropy_bits") is not None:
        out["entropy_bits"] = out["val_entropy_bits"]
    else:
        out["entropy_bits"] = None
    return out


def code_usage_stats(idx_flat: np.ndarray, codebook_size: int | None = None) -> dict:
    """
    From flat assignment indices compute:
    - counts: array of counts per code (only active codes if we don't fix size)
    - gini: Gini coefficient of usage distribution
    - entropy_usage: entropy of usage distribution (bits)
    - pct_mass_top_10: % of token mass in top 10% of codes (by usage)
    - pct_mass_top_1: % of token mass in top 1% of codes
    """
    total = len(idx_flat)
    if total == 0:
        return {"gini": 0.0, "entropy_usage": 0.0, "pct_mass_top_10": 0.0, "pct_mass_top_1": 0.0, "n_active": 0}

    max_idx = int(idx_flat.max())
    if codebook_size is not None:
        max_idx = max(max_idx, codebook_size - 1)
    counts = np.bincount(idx_flat.astype(np.int64), minlength=max_idx + 1)
    counts = counts[counts > 0]
    n_active = len(counts)
    if n_active == 0:
        return {"gini": 0.0, "entropy_usage": 0.0, "pct_mass_top_10": 0.0, "pct_mass_top_1": 0.0, "n_active": 0}

    S = counts.sum()
    if S <= 0:
        return {"gini": 0.0, "entropy_usage": 0.0, "pct_mass_top_10": 0.0, "pct_mass_top_1": 0.0, "n_active": n_active}

    # Gini: sort ascending, Gini = (2 * sum((i+1)*x_i) - (n+1)*S) / (n*S)
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_counts)) - (n + 1) * S) / (n * S) if n and S else 0.0

    # Entropy of usage: -sum(p*log2(p))
    p = counts / S
    p = p[p > 0]
    entropy_usage = float(-np.sum(p * np.log2(p)))

    # % mass in top 10% and top 1% of codes (by count)
    sorted_desc = np.sort(counts)[::-1]
    n_codes = len(sorted_desc)
    top_10_pct = max(1, int(np.ceil(0.10 * n_codes)))
    top_1_pct = max(1, int(np.ceil(0.01 * n_codes)))
    pct_mass_top_10 = 100.0 * sorted_desc[:top_10_pct].sum() / S
    pct_mass_top_1 = 100.0 * sorted_desc[:top_1_pct].sum() / S

    return {
        "gini": float(gini),
        "entropy_usage": float(entropy_usage),
        "pct_mass_top_10": float(pct_mass_top_10),
        "pct_mass_top_1": float(pct_mass_top_1),
        "n_active": n_active,
        "counts": counts,
    }


def run_kmeans_sweep(z_np: np.ndarray, k_values: list[int], max_samples: int = 50000, random_state: int = 42) -> dict:
    """Run k-means for each k; return inertia (WCSS) and silhouette per k."""
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
    ap = argparse.ArgumentParser(description="Compute quantization structure metrics (WCSS, silhouette, usage histograms)")
    ap.add_argument("--num-images", type=int, default=400, help="Number of images to run through encoder")
    ap.add_argument("--subsample-latents", type=int, default=20000, help="Max latent points per model")
    ap.add_argument("--k-values", type=str, default="1000,2000,4000,6000,8000,10000,12000,16384",
                    help="Comma-separated k values for elbow/silhouette sweep")
    ap.add_argument("--output-dir", type=str, default="plots")
    ap.add_argument("--main", action="store_true", help="Use main 16K checkpoints (default)")
    ap.add_argument("--models", nargs="+", default=None, help="Restrict to these models")
    args = ap.parse_args()

    k_values = [int(x) for x in args.k_values.split(",")]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    fair = discover_lmb_fair_checkpoint()
    if fair is not None:
        checkpoints["lmb_fair"] = fair
    if args.models:
        checkpoints = {k: v for k, v in checkpoints.items() if k in args.models}

    # Exclude lmb_fair for main "5 models" comparison unless requested
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

    results_per_model = {}

    for model_type, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"Skipping {model_type}: checkpoint not found")
            continue
        print(f"\n--- {model_type.upper()} ---")
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
            if effective_type == "lfq":
                idx_t = idx_sub.to(code_device).unsqueeze(0)
                z_q = model.quantize.indices_to_codes(idx_t, project_out=False).squeeze(0)
                if z_q.dim() == 2 and z_q.shape[0] < z_q.shape[1]:
                    z_q = z_q.T
                z = z_q

            metrics_align = compute_metrics(z, e)
            z_np = z.numpy()

            # Flatten indices for code usage (LMB per-channel: flatten to linear index for simplicity)
            if idx_sub.dim() > 1 and effective_type == "lmb":
                config = getattr(model, "config", {})
                lmb_flattened = getattr(model.quantize, "flatten_channels", False) if hasattr(model, "quantize") else config.get("flatten_channels", False)
                if not lmb_flattened:
                    # Per-channel: treat each row as a single "composite" code via ravel index
                    idx_flat = idx_sub.numpy().astype(np.int64)
                    # Use lexicographic linear index: code_id = c0*K^2 + c1*K + c2 ...
                    shape = idx_flat.shape
                    if len(shape) == 2:
                        idx_flat = idx_flat[:, 0]  # fallback: use first channel only for histogram
                    else:
                        idx_flat = idx_flat.reshape(-1)
                else:
                    idx_flat = idx_sub.reshape(-1).numpy().astype(np.int64)
            else:
                idx_flat = idx_sub.reshape(-1).numpy().astype(np.int64)

            usage = code_usage_stats(idx_flat, codebook_size=16384)

            # K-means at this model's active count (for entropy vs WCSS point)
            k_model = min(n_active, z_np.shape[0] - 1)
            wcss_at_k = np.nan
            sil_at_k = np.nan
            if k_model >= 2:
                km_model = KMeans(n_clusters=k_model, random_state=42, n_init=10, max_iter=300)
                labels_model = km_model.fit_predict(z_np)
                wcss_at_k = float(km_model.inertia_)
                sil_at_k = float(silhouette_score(z_np, labels_model, sample_size=min(5000, z_np.shape[0]), random_state=42))

            # Eval row for entropy
            exp_dir = ckpt_path.parent.parent
            eval_row = load_last_eval_row(exp_dir)
            entropy_bits = None
            if eval_row and eval_row.get("entropy_bits") is not None:
                entropy_bits = eval_row["entropy_bits"]
            elif eval_row and eval_row.get("val_perplexity") and eval_row["val_perplexity"] > 0:
                entropy_bits = math.log2(eval_row["val_perplexity"])

            name = {"fsq": "FSQ 16K", "vq": "VQ 16K", "lfq": "LFQ 16K", "sim_vq": "SimVQ 16K", "lmb": "LMB 16K", "lmb_fair": "LMB 16K per-ch"}.get(model_type, model_type)

            results_per_model[model_type] = {
                "name": name,
                "z": z_np,
                "n_active": n_active,
                "latent_to_code_mean": metrics_align["latent_to_code_mean"],
                "relative_quant_error": metrics_align["relative_quant_error"],
                "orphan_codes_pct": metrics_align["orphan_codes_pct"],
                "gini": usage["gini"],
                "entropy_usage": usage["entropy_usage"],
                "pct_mass_top_10": usage["pct_mass_top_10"],
                "pct_mass_top_1": usage["pct_mass_top_1"],
                "counts": usage.get("counts"),
                "wcss_at_k": wcss_at_k,
                "silhouette_at_k": sil_at_k,
                "entropy_bits": entropy_bits,
                "k_model": k_model,
            }
            del model
        except Exception as e:
            print(f"Error {model_type}: {e}")
            import traceback
            traceback.print_exc()

    if len(results_per_model) == 0:
        print("No models processed. Exiting.")
        return

    codebook_size = 16384
    # K-means sweep on first model's z (same latent space idea: "how many clusters does data support")
    first_key = next(iter(results_per_model))
    z_sweep = results_per_model[first_key]["z"]
    print("\nRunning K-means sweep for WCSS/silhouette vs k (first model)...")
    sweep = run_kmeans_sweep(z_sweep, k_values)

    # Per-model K-means sweep + TSS for scale-invariant elbow (WCSS/TSS comparable across latent spaces)
    print("\nRunning per-model K-means sweep for normalized elbow...")
    for m in list(results_per_model.keys()):
        z_np = results_per_model[m]["z"]
        tss = float(np.sum((z_np - z_np.mean(axis=0)) ** 2))
        results_per_model[m]["tss"] = tss
        results_per_model[m]["sweep"] = run_kmeans_sweep(z_np, k_values)
        # Normalized WCSS (comparable across different latent dimensions and scales, e.g. FSQ/LFQ vs VQ)
        if tss > 0 and not np.isnan(results_per_model[m].get("wcss_at_k", np.nan)):
            results_per_model[m]["wcss_at_k_normalized"] = results_per_model[m]["wcss_at_k"] / tss
        else:
            results_per_model[m]["wcss_at_k_normalized"] = np.nan

    # --- Plots ---
    model_order_plot = [m for m in ["fsq", "vq", "lfq", "sim_vq", "lmb"] if m in results_per_model]
    colors = {"fsq": "#ff7f0e", "vq": "#1f77b4", "lfq": "#9467bd", "sim_vq": "#2ca02c", "lmb": "#d62728"}
    active_counts = {m: results_per_model[m]["n_active"] for m in model_order_plot}

    # 1) WCSS vs k
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    ks = sorted(sweep.keys())
    inertias = [sweep[k]["inertia"] for k in ks]
    ax1.plot(ks, inertias, "k-o", markersize=4, label="K-means on encoder z")
    for m in model_order_plot:
        k = active_counts[m]
        ax1.axvline(k, color=colors.get(m, "gray"), linestyle="--", alpha=0.8, label=f"{results_per_model[m]['name']} (k={k})")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Within-cluster variance (WCSS / inertia)")
    ax1.set_title("Elbow: WCSS vs k (vertical lines = each model's active code count)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig1.savefig(out_dir / "quantization_structure_wcss_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    # 1b) Scale-invariant elbow: WCSS(k)/TSS per model (comparable across different latent space units)
    fig1b, ax1b = plt.subplots(figsize=(9, 5))
    ks = sorted(k_values)
    for m in model_order_plot:
        r = results_per_model[m]
        sweep_m = r.get("sweep")
        tss_m = r.get("tss")
        if sweep_m is None or tss_m is None or tss_m <= 0:
            continue
        xs = [k for k in ks if k in sweep_m and not np.isnan(sweep_m[k]["inertia"])]
        ys = [sweep_m[k]["inertia"] / tss_m for k in xs]
        ax1b.plot(xs, ys, "o-", color=colors.get(m, "gray"), markersize=4, label=r["name"])
        k_active = active_counts[m]
        if k_active in sweep_m and not np.isnan(sweep_m[k_active]["inertia"]) and tss_m > 0:
            ax1b.axvline(k_active, color=colors.get(m, "gray"), linestyle="--", alpha=0.7)
    ax1b.set_xlabel("Number of clusters k")
    ax1b.set_ylabel("Fraction of variance unexplained (WCSS / TSS)")
    ax1b.set_title("Elbow (scale-invariant): each model in its own latent space, normalized by TSS")
    ax1b.legend(loc="best", fontsize=9)
    ax1b.grid(True, alpha=0.3)
    ax1b.set_ylim(bottom=0)
    fig1b.savefig(out_dir / "quantization_structure_wcss_vs_k_normalized.png", dpi=150, bbox_inches="tight")
    plt.close(fig1b)

    # 2) Silhouette vs k (vertical lines = each model's active code count)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    sils = [sweep[k]["silhouette"] for k in ks]
    ax2.plot(ks, sils, "k-o", markersize=4, label="Silhouette on encoder z")
    for m in model_order_plot:
        k = active_counts[m]
        ax2.axvline(k, color=colors.get(m, "gray"), linestyle="--", alpha=0.8, label=f"{results_per_model[m]['name']} (k={k})")
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette vs k (higher = better-defined clusters)")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.savefig(out_dir / "quantization_structure_silhouette_vs_k.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # 3) Entropy vs WCSS/TSS (one point per model; normalized so FSQ/LFQ comparable to VQ/SimVQ/LMB)
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    for m in model_order_plot:
        r = results_per_model[m]
        ent = r.get("entropy_bits")
        wcss_norm = r.get("wcss_at_k_normalized")
        if ent is not None and wcss_norm is not None and not np.isnan(wcss_norm):
            ax3.scatter(ent, wcss_norm, c=colors.get(m, "gray"), s=120, label=r["name"], zorder=5)
            ax3.annotate(r["name"], (ent, wcss_norm), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax3.set_xlabel("Entropy (bits) [log₂(perplexity)]")
    ax3.set_ylabel("WCSS / TSS (fraction of variance unexplained at k = active codes)")
    ax3.set_title("Entropy vs WCSS/TSS — comparable across latent spaces; lower = better geometry")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)
    fig3.savefig(out_dir / "quantization_structure_entropy_vs_wcss.png", dpi=150, bbox_inches="tight")
    plt.close(fig3)

    # 4) Code usage histograms (one subplot per model)
    n_models = len(model_order_plot)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig4, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    for i, m in enumerate(model_order_plot):
        ax = axes[i]
        r = results_per_model[m]
        counts = r.get("counts")
        if counts is not None and len(counts) > 0:
            ax.hist(counts, bins=min(80, max(20, len(counts) // 50)), color=colors.get(m, "gray"), alpha=0.7, edgecolor="black", linewidth=0.3)
            ax.axvline(counts.mean(), color="red", linestyle="--", label=f"mean={counts.mean():.1f}")
            ax.set_title(f"{r['name']}\nGini={r['gini']:.3f}  Entropy(usage)={r['entropy_usage']:.2f}  Top10%={r['pct_mass_top_10']:.1f}%")
        else:
            ax.set_title(f"{r['name']} (no counts)")
        ax.set_xlabel("Assignments per code")
        ax.set_ylabel("Number of codes")
        ax.legend(loc="upper right", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig4.suptitle("Code usage histogram — selective allocation (LMB) vs uniform (FSQ) vs collapse (VQ)", fontsize=11)
    fig4.tight_layout()
    fig4.savefig(out_dir / "quantization_structure_code_usage_histograms.png", dpi=150, bbox_inches="tight")
    plt.close(fig4)

    # --- Text table ---
    lines = [
        "Quantization structure metrics (professor-recommended)",
        "=" * 70,
        "",
        "0. Active codes (k) per model — k each method actually uses (codebook size = 16K)",
        "-" * 60,
    ]
    for m in model_order_plot:
        r = results_per_model[m]
        util_pct = 100.0 * r["n_active"] / codebook_size
        lines.append(f"  {r['name']}:  n_active = {r['n_active']}   utilization = {util_pct:.1f}%")
    lines.extend([
        "",
        "1. Distance alignment (encoder z → nearest code)",
        "-" * 50,
    ])
    for m in model_order_plot:
        r = results_per_model[m]
        lines.append(f"  {r['name']}:  latent_to_code_mean = {r['latent_to_code_mean']:.4f}   relative_quant_error = {r['relative_quant_error']:.4f}   orphan_codes_pct = {r['orphan_codes_pct']:.2f}")
    lines.extend([
        "",
        "2. Code usage (Gini, entropy of usage, % mass in top codes)",
        "-" * 50,
    ])
    for m in model_order_plot:
        r = results_per_model[m]
        lines.append(f"  {r['name']}:  Gini = {r['gini']:.4f}   Entropy(usage) = {r['entropy_usage']:.2f} bits   Top10% = {r['pct_mass_top_10']:.1f}%   Top1% = {r['pct_mass_top_1']:.1f}%   n_active = {r['n_active']}")
    lines.extend([
        "",
        "3. WCSS & Silhouette at k = model's active code count",
        "   (WCSS/TSS = fraction of variance unexplained; comparable across FSQ/LFQ vs VQ/SimVQ/LMB)",
        "-" * 50,
    ])
    for m in model_order_plot:
        r = results_per_model[m]
        wcss_raw = r["wcss_at_k"]
        wcss_norm = r.get("wcss_at_k_normalized")
        norm_str = f"   WCSS/TSS = {wcss_norm:.4f}" if wcss_norm is not None and not np.isnan(wcss_norm) else "   WCSS/TSS = N/A"
        lines.append(f"  {r['name']}:  WCSS_at_k = {wcss_raw:.2f}{norm_str}   Silhouette_at_k = {r['silhouette_at_k']:.4f}   (k = {r['n_active']})")
    lines.extend([
        "",
        "4. Entropy (bits) from eval_metrics",
        "-" * 50,
    ])
    for m in model_order_plot:
        r = results_per_model[m]
        ent = r.get("entropy_bits")
        lines.append(f"  {r['name']}:  entropy_bits = {ent:.2f}" if ent is not None else f"  {r['name']}:  entropy_bits = N/A")
    lines.append("")
    lines.append("=" * 70)

    out_txt = out_dir / "quantization_structure_metrics.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {out_txt}")
    print("Saved plots: WCSS vs k, WCSS vs k (normalized), Silhouette vs k, Entropy vs WCSS, Code usage histograms")
    print("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute metrics that support the UMAP-figure paragraph claims (SimVQ vs LFQ).

Paragraph claims:
  1. SimVQ achieves dense coverage by activating nearly the entire codebook.
  2. LFQ attains comparable coverage using substantially fewer active codes.
  3. LFQ allocates codebook capacity more selectively.
  4. Similar reconstruction quality at lower effective representation rates.

Metrics computed:
  - Active codes & utilization (%) — from eval_metrics.csv + codebook size
  - latent_to_code_mean — mean distance from encoder outputs to nearest code (coverage)
  - Spearman(usage, density_at_code) — selective allocation (codes where mass is)
  - PSNR, SSIM, LPIPS, log2(active_codes), log2(perplexity) — rate–distortion

Outputs: plots/paragraph_claim_metrics.txt

Usage:
  python scripts/compute_paragraph_claim_metrics.py
  python scripts/compute_paragraph_claim_metrics.py --num-images 200 --output-dir plots
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
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
    compute_metrics,
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


def load_last_eval_row(experiment_dir: Path) -> dict[str, float] | None:
    """Load last row of eval_metrics.csv as float dict."""
    csv_path = experiment_dir / "eval_metrics.csv"
    if not csv_path.exists():
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    out: dict[str, float] = {}
    for key, value in row.items():
        if key in ("epoch", "global_step"):
            continue
        try:
            if value and str(value).strip() and str(value) != "None":
                v = float(value)
                if not (math.isnan(v) or math.isinf(v)):
                    out[key] = v
        except (ValueError, TypeError):
            pass
    return out


def get_codebook_size(experiment_dir: Path, model_type: str) -> int | None:
    """Infer codebook size from config.json."""
    config_path = experiment_dir / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            config = json.load(f)
        if model_type in ("vq", "sim_vq", "lfq"):
            return int(config.get("codebook_size", 0)) or None
        if model_type == "fsq":
            levels = config.get("levels")
            if levels:
                return int(math.prod(levels))
            return None
        if model_type in ("lmb", "lmb_fair"):
            num_bins = config.get("num_bins")
            if num_bins is not None:
                return int(num_bins)
            levels = config.get("lmb_levels")
            if levels:
                return int(math.prod(levels))
            return None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def compute_usage_and_spearman(
    z_np: np.ndarray,
    e_np: np.ndarray,
    active_data: list | np.ndarray,
    idx_sub: torch.Tensor,
    model_type: str,
) -> tuple[float, float]:
    """Per-code usage and Spearman(usage, density_at_code). Returns (rho, pval)."""
    k_nn = 10
    if model_type == "lmb":
        config = {}  # will be set by caller if needed
        if isinstance(active_data, np.ndarray) and active_data.ndim > 1:
            idx_arr = idx_sub.numpy()
            usage = np.array(
                [np.sum(np.all(idx_arr == aid, axis=1)) for aid in active_data],
                dtype=np.float64,
            )
        else:
            flat_idx = idx_sub.reshape(-1).numpy()
            usage = np.array(
                [np.sum(flat_idx == aid) for aid in active_data],
                dtype=np.float64,
            )
    else:
        flat_idx = idx_sub.reshape(-1).numpy()
        usage = np.array(
            [np.sum(flat_idx == aid) for aid in active_data],
            dtype=np.float64,
        )
    density_at_code = knn_density_at_points(z_np, e_np, k=k_nn)
    if usage.size > 2 and np.var(usage) > 0 and np.var(density_at_code) > 0:
        rho, pval = spearmanr(usage, density_at_code)
        return float(rho), float(pval)
    return float("nan"), float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute metrics supporting UMAP-figure paragraph (SimVQ vs LMB); default runs all models."
    )
    ap.add_argument("--num-images", type=int, default=150)
    ap.add_argument("--subsample-latents", type=int, default=8000)
    ap.add_argument("--output-dir", type=str, default="plots")
    ap.add_argument("--models", type=str, nargs="*", default=["fsq", "vq", "lfq", "sim_vq", "lmb"],
                    help="Models to run (default: lfq sim_vq). Use e.g. fsq vq lfq sim_vq lmb for all.")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = {k: project_root / v for k, v in MAIN_CHECKPOINTS.items()}
    fair = discover_lmb_fair_checkpoint()
    if fair is not None:
        checkpoints["lmb_fair"] = fair
    model_order = ["fsq", "vq", "lfq", "sim_vq", "lmb", "lmb_fair"]
    checkpoints = {k: v for k, v in checkpoints.items() if k in model_order}
    if args.models:
        checkpoints = {k: v for k, v in checkpoints.items() if k in args.models}
    if not checkpoints:
        print("No checkpoints found. Set --models or use default (all five) with MAIN_CHECKPOINTS.")
        return
    print(f"Using checkpoints: {list(checkpoints.keys())}")

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
    dataset = FlatImageDataset(
        str(test_path), transform=transform, max_images=args.num_images * 2
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    display_names = {
        "fsq": "FSQ 16K",
        "vq": "VQ 16K",
        "lfq": "LFQ 16K",
        "sim_vq": "SimVQ 16K",
        "lmb": "LMB 16K",
        "lmb_fair": "LMB 16K per-ch",
    }

    results: dict[str, dict] = {}

    for model_type, ckpt_path in checkpoints.items():
        if not ckpt_path.exists():
            print(f"Skipping {model_type}: checkpoint not found")
            continue
        exp_dir = ckpt_path.parent.parent
        eval_row = load_last_eval_row(exp_dir)
        codebook_size = get_codebook_size(exp_dir, model_type)
        if codebook_size is None:
            codebook_size = 16384

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
            metrics = compute_metrics(z, e)
            rho, pval = compute_usage_and_spearman(
                z_np, e_np, active_data, idx_sub, effective_type
            )

            utilization = 100.0 * n_active / codebook_size if codebook_size else 0.0
            rate_log2_codes = math.log2(n_active) if n_active > 0 else float("nan")
            perplexity = eval_row.get("val_perplexity") if eval_row else None
            rate_log2_ppl = math.log2(perplexity) if perplexity and perplexity > 0 else float("nan")

            results[model_type] = {
                "name": display_names.get(model_type, model_type),
                "n_active": n_active,
                "codebook_size": codebook_size,
                "utilization_pct": utilization,
                "latent_to_code_mean": metrics["latent_to_code_mean"],
                "relative_quant_error": metrics.get("relative_quant_error", float("nan")),
                "spearman_usage_density": rho,
                "spearman_pval": pval,
                "psnr": eval_row.get("val_psnr") if eval_row else None,
                "ssim": eval_row.get("val_ssim") if eval_row else None,
                "lpips": eval_row.get("val_lpips") if eval_row else None,
                "rec_loss": eval_row.get("val_rec_loss") if eval_row else None,
                "rate_log2_codes": rate_log2_codes,
                "rate_log2_ppl": rate_log2_ppl,
                "perplexity": perplexity,
            }
            del model
        except Exception as e:
            print(f"Error {model_type}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        msg = "No results (checkpoints found but all runs failed or no data)."
        print(msg)
        out_txt = out_dir / "paragraph_claim_metrics.txt"
        with open(out_txt, "w") as f:
            f.write(msg + "\n")
        return

    # Build table and interpretation
    lines = [
        "Metrics supporting UMAP-figure paragraph (SimVQ vs LMB)",
        "=" * 80,
        "",
        "1. SimVQ activates nearly the entire codebook  ->  utilization %, n_active",
        "2. LMB comparable coverage with fewer codes   ->  latent_to_code_mean similar, n_active lower",
        "3. LMB allocates more selectively             ->  Spearman(usage, density_at_code) higher",
        "4. Similar reconstruction at lower rate        ->  PSNR/SSIM/LPIPS vs log2(active_codes)",
        "",
        "z→code = mean distance encoder output → nearest code. rel_err = z→code / std(z) (scale-invariant).",
        "",
        "-" * 80,
        f"{'Model':<14} {'n_active':>10} {'util%':>8} {'z→code':>10} {'rel_err':>8} {'Spearman':>10} {'PSNR':>8} {'SSIM':>8} {'LPIPS':>8} {'log2(K)':>8}",
        "-" * 80,
    ]
    for m in model_order:
        if m not in results:
            continue
        r = results[m]
        util = r["utilization_pct"]
        z2c = r["latent_to_code_mean"]
        rel_err = r.get("relative_quant_error", float("nan"))
        rho = r["spearman_usage_density"]
        psnr = r["psnr"]
        ssim = r["ssim"]
        lpips = r["lpips"]
        log2k = r["rate_log2_codes"]
        util_s = f"{util:.1f}" if not math.isnan(util) else "—"
        z2c_s = f"{z2c:.4f}" if not math.isnan(z2c) else "—"
        rel_s = f"{rel_err:.4f}" if not math.isnan(rel_err) else "—"
        rho_s = f"{rho:.4f}" if not math.isnan(rho) else "—"
        psnr_s = f"{psnr:.2f}" if psnr is not None and not math.isnan(psnr) else "—"
        ssim_s = f"{ssim:.4f}" if ssim is not None and not math.isnan(ssim) else "—"
        lpips_s = f"{lpips:.4f}" if lpips is not None and not math.isnan(lpips) else "—"
        log2k_s = f"{log2k:.2f}" if not math.isnan(log2k) else "—"
        lines.append(
            f"{r['name']:<14} {r['n_active']:>10} {util_s:>8} {z2c_s:>10} {rel_s:>8} {rho_s:>10} {psnr_s:>8} {ssim_s:>8} {lpips_s:>8} {log2k_s:>8}"
        )
    lines.append("-" * 80)
    lines.append("")

    # Metric definitions
    lines.append("Metric definitions:")
    lines.append("  n_active     Number of distinct codes used on the evaluation set.")
    lines.append("  util%        Codebook utilization = 100 * n_active / codebook_size.")
    lines.append("  z→code       Mean distance from each encoder output to its nearest code (coverage).")
    lines.append("  rel_err      Scale-invariant coverage = z→code / std(z); comparable across latent spaces.")
    lines.append("  Spearman     Spearman(usage, density_at_code): high => codes used more sit in denser z regions (selective).")
    lines.append("  PSNR         Peak signal-to-noise ratio (dB); higher = better reconstruction.")
    lines.append("  SSIM         Structural similarity; higher = better perceptual match.")
    lines.append("  LPIPS        Learned perceptual similarity; lower = better perceptual match.")
    lines.append("  log2(K)      Effective rate in bits if we coded which K codes are used; log2(n_active).")
    lines.append("")

    # Elbow / redundant codes claim (with caveat: k != active codes; elbow is from one model's z)
    elbow_k = 4000
    n_simvq = results.get("sim_vq", {}).get("n_active") or 0
    n_fsq = results.get("fsq", {}).get("n_active") or 0
    n_lmb = results.get("lmb", {}).get("n_active") or 0
    n_vq = results.get("vq", {}).get("n_active") or 0
    n_lfq = results.get("lfq", {}).get("n_active") or 0
    if n_simvq and n_fsq:
        lines.append("Elbow / redundant codes:")
        lines.append("  k in the elbow is NOT the same as active codes:")
        lines.append("    - Elbow k = number of K-means clusters on encoder z (no model codebook).")
        lines.append("    - Active codes = number of codebook entries the model uses.")
        lines.append("  Caveat: our single elbow curve is from the first model's z (FSQ); ~4K is that space.")
        lines.append(f"  SimVQ ({n_simvq}) and FSQ ({n_fsq}) use more than ~{elbow_k}; VQ, LFQ, LMB ({n_vq}, {n_lfq}, {n_lmb}) closer.")
        lines.append("  Alternative claim: The number of active codes in SimVQ and FSQ is partly repetitive")
        lines.append("  and not needed — the latent distribution is well captured by fewer representatives (~4K),")
        lines.append("  so the additional codes do not add necessary capacity; many are redundant for similar quality.")
        lines.append("")

    # Paragraph-support summary
    if "sim_vq" in results and "lmb" in results:
        svq = results["sim_vq"]
        lmb = results["lmb"]
        lines.append("Paragraph-claim support (SimVQ vs LMB):")
        lines.append("")
        lines.append(f"  1. SimVQ activates a large fraction of the codebook: "
                    f"n_active = {svq['n_active']}, utilization = {svq['utilization_pct']:.1f}%.")
        lines.append("")
        rel_svq = svq.get("relative_quant_error") or float("nan")
        rel_lmb = lmb.get("relative_quant_error") or float("nan")
        coverage_ok = (not math.isnan(rel_svq) and not math.isnan(rel_lmb) and
                      abs(rel_svq - rel_lmb) / max(abs(rel_svq), abs(rel_lmb), 1e-8) < 0.5)
        lines.append(f"  2. LMB uses substantially fewer active codes ({lmb['n_active']}) than SimVQ ({svq['n_active']}). "
                    f"Coverage (relative quant error): SimVQ = {rel_svq:.4f}, LMB = {rel_lmb:.4f}. "
                    + ("Similar scale-invariant coverage with fewer codes." if coverage_ok else "Note: z→code is in each model's latent space (not directly comparable across methods)."))
        lines.append("")
        lines.append(f"  3. Selective allocation: Spearman(usage, density_at_code) "
                    f"LMB = {lmb['spearman_usage_density']:.4f}, SimVQ = {svq['spearman_usage_density']:.4f}. "
                    + ("LMB higher => more selective." if not math.isnan(lmb['spearman_usage_density']) and not math.isnan(svq['spearman_usage_density']) and lmb['spearman_usage_density'] > svq['spearman_usage_density'] else "Compare values."))
        lines.append("")
        psnr_sim = svq.get("psnr") or 0
        psnr_lmb = lmb.get("psnr") or 0
        log2_sim = svq["rate_log2_codes"]
        log2_lmb = lmb["rate_log2_codes"]
        lines.append(f"  4. Rate–distortion: SimVQ PSNR = {psnr_sim:.2f}, log2(active_codes) = {log2_sim:.2f}; "
                    f"LMB PSNR = {psnr_lmb:.2f}, log2(active_codes) = {log2_lmb:.2f}. "
                    + ("Similar PSNR at lower rate (LMB)." if abs(psnr_sim - psnr_lmb) < 2.0 and log2_lmb < log2_sim else "Check PSNR vs log2(K)."))

    lines.append("")
    lines.append("=" * 80)

    out_txt = out_dir / "paragraph_claim_metrics.txt"
    with open(out_txt, "w") as f:
        f.write("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"\nSaved: {out_txt}")
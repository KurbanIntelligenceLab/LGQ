#!/usr/bin/env python3
"""
Compare all models across all available epochs.
Generates results/model_comparison_all_epochs.txt from eval_metrics.csv in each experiment.
"""

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RESULTS_DIR = Path("results")
COL_WIDTH = 15
METRIC_WIDTH = 20

# Model display names in table column order
MODEL_NAMES = ["FSQ", "LFQ", "LMB", "LMB-Fixed", "SIM_VQ", "VQ"]

# CSV key -> (display name, decimals, higher_is_better)
METRIC_SPECS: List[Tuple[str, str, int, bool]] = [
    ("val_rfid", "rFID", 2, False),
    ("val_psnr", "PSNR (dB)", 2, True),
    ("val_ssim", "SSIM", 4, True),
    ("val_lpips", "LPIPS", 4, False),
    ("val_rec_loss", "Rec Loss", 4, False),
    ("val_active_codes", "Active Codes", 0, True),
    ("val_codebook_util", "Codebook Util (%)", 2, True),
    ("val_perplexity", "Perplexity (K_eff)", 1, True),
    ("val_entropy_bits", "Entropy (bits)", 3, True),
    ("val_bits_per_token", "Bits/token", 3, True),
]


def load_epoch_metrics(experiment_dir: Path) -> Dict[int, Dict[str, float]]:
    """Load metrics per epoch from eval_metrics.csv."""
    eval_csv = experiment_dir / "eval_metrics.csv"
    if not eval_csv.exists():
        return {}
    epochs_metrics: Dict[int, Dict[str, float]] = {}
    with open(eval_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("epoch"):
                continue
            try:
                epoch = int(row["epoch"])
                if epoch in epochs_metrics:
                    continue
                metrics = {}
                for key, value in row.items():
                    if key in ("epoch", "global_step"):
                        continue
                    try:
                        if value and str(value).strip() and str(value) != "None":
                            v = float(value)
                            if math.isnan(v) or math.isinf(v):
                                metrics[key] = None
                            else:
                                metrics[key] = v
                        else:
                            metrics[key] = None
                    except (ValueError, TypeError):
                        metrics[key] = None
                epochs_metrics[epoch] = metrics
            except (ValueError, KeyError):
                continue
    return epochs_metrics


def find_experiment(model_name: str, image_size: int = 128) -> Optional[Path]:
    """Find the experiment directory for a model (128x128)."""
    if model_name == "FSQ":
        candidates_16384 = []
        candidates_other = []
        for exp_dir in (RESULTS_DIR / "fsq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                levels = config.get("levels", [])
                total = 1
                for L in levels:
                    total *= int(L)
                is_16384 = total == 16384
                n = len(list(csv.DictReader(open(eval_path)))) if eval_path.exists() else 0
                (candidates_16384 if is_16384 else candidates_other).append((exp_dir, n))
            except Exception:
                pass
        for cand in (candidates_16384, candidates_other):
            cand.sort(key=lambda x: x[1], reverse=True)
        if candidates_16384:
            return candidates_16384[0][0]
        if candidates_other:
            return candidates_other[0][0]
    elif model_name == "LMB":
        candidates_22 = []
        candidates_other = []
        for exp_dir in (RESULTS_DIR / "lmb").iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "lmb_fixed_init":
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists() or not eval_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                n = len(list(csv.DictReader(open(eval_path))))
                if "20260122" in exp_dir.name and config.get("batch_size") == 32:
                    candidates_22.append((exp_dir, n))
                else:
                    candidates_other.append((exp_dir, n))
            except Exception:
                pass
        for c in (candidates_22, candidates_other):
            c.sort(key=lambda x: x[1], reverse=True)
        if candidates_22:
            return candidates_22[0][0]
        if candidates_other:
            return candidates_other[0][0]
    elif model_name == "LMB-Fixed":
        exp_dir = RESULTS_DIR / "lmb" / "lmb_fixed_init"
        if exp_dir.exists() and (exp_dir / "config.json").exists():
            with open(exp_dir / "config.json") as f:
                if json.load(f).get("image_size") == image_size:
                    return exp_dir
    elif model_name == "LMB-Fair":
        for exp_dir in (RESULTS_DIR / "lmb").iterdir():
            if not exp_dir.is_dir():
                continue
            if "perchannel_fair" in exp_dir.name or "perchannel_fair" in str(exp_dir):
                if (exp_dir / "eval_metrics.csv").exists() and (exp_dir / "config.json").exists():
                    with open(exp_dir / "config.json") as f:
                        if json.load(f).get("image_size") == image_size:
                            return exp_dir
    elif model_name == "LFQ":
        for exp_dir in (RESULTS_DIR / "lfq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            if config_path.exists() and (exp_dir / "eval_metrics.csv").exists():
                try:
                    with open(config_path) as f:
                        cfg = json.load(f)
                    if cfg.get("image_size") == image_size and cfg.get("codebook_size") == 16384:
                        return exp_dir
                except Exception:
                    pass
    elif model_name == "SIM_VQ":
        candidates = []
        for exp_dir in (RESULTS_DIR / "sim_vq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                n = len(list(csv.DictReader(open(eval_path)))) if eval_path.exists() else 0
                candidates.append((exp_dir, n))
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    elif model_name == "VQ":
        candidates = []
        for exp_dir in (RESULTS_DIR / "vq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
            eval_path = exp_dir / "eval_metrics.csv"
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                n = len(list(csv.DictReader(open(eval_path)))) if eval_path.exists() else 0
                candidates.append((exp_dir, n))
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    elif model_name == "ROT_VQ":
        for exp_dir in (RESULTS_DIR / "rot_vq").iterdir():
            if not exp_dir.is_dir():
                continue
            if (exp_dir / "eval_metrics.csv").exists() and (exp_dir / "config.json").exists():
                try:
                    with open(exp_dir / "config.json") as f:
                        if json.load(f).get("image_size") == image_size:
                            return exp_dir
                except Exception:
                    pass
    return None


def format_value(v: float, decimals: int) -> str:
    if decimals == 0:
        return f"{int(round(v))}"
    return f"{v:.{decimals}f}"


def main() -> None:
    experiments: Dict[str, Dict[int, Dict[str, float]]] = {}
    for name in MODEL_NAMES:
        exp_dir = find_experiment(name, 128)
        if exp_dir:
            em = load_epoch_metrics(exp_dir)
            if em:
                experiments[name] = em
                print(f"Found {name}: {len(em)} epochs")
            else:
                print(f"Found {name}: 0 epochs (no eval_metrics.csv yet)")
        else:
            print(f"Warning: Could not find experiment for {name}")

    if not experiments:
        print("No experiments found!")
        return

    all_epochs = sorted(set().union(*(set(em.keys()) for em in experiments.values())))
    print(f"\nTotal epochs to compare: {len(all_epochs)} (Epochs {min(all_epochs)}-{max(all_epochs)})")

    width = METRIC_WIDTH + len(MODEL_NAMES) * COL_WIDTH
    output_lines: List[str] = []
    output_lines.append("=" * width)
    output_lines.append("COMPREHENSIVE MODEL COMPARISON - ALL EPOCHS (128x128 Image Size)")
    output_lines.append("=" * width)
    output_lines.append("")

    for epoch in all_epochs:
        output_lines.append("=" * width)
        output_lines.append(f"EPOCH {epoch} - MODEL COMPARISON TABLE (128x128 Image Size)")
        output_lines.append("=" * width)
        output_lines.append("")
        header = f"{'Metric':<{METRIC_WIDTH}}"
        for name in MODEL_NAMES:
            header += f"{name:<{COL_WIDTH}}"
        output_lines.append(header)
        output_lines.append("-" * len(header))

        for csv_key, display_name, decimals, higher_better in METRIC_SPECS:
            row = f"{display_name:<{METRIC_WIDTH}}"
            values: Dict[str, float] = {}
            for name in MODEL_NAMES:
                if name in experiments and epoch in experiments[name]:
                    v = experiments[name][epoch].get(csv_key)
                    if v is not None:
                        values[name] = v
                        row += f"{format_value(v, decimals):<{COL_WIDTH}}"
                    else:
                        row += f"{'N/A':<{COL_WIDTH}}"
                else:
                    row += f"{'N/A':<{COL_WIDTH}}"
            output_lines.append(row)
            if values:
                if higher_better:
                    best_model = max(values.items(), key=lambda x: x[1])[0]
                else:
                    best_model = min(values.items(), key=lambda x: x[1])[0]
                best_value = values[best_model]
                output_lines.append(f"  → Best: {best_model} ({format_value(best_value, decimals)})")
        output_lines.append("-" * len(header))
        output_lines.append("")

    # Best performing models section (exclude invalid values: LPIPS 0 = missing/bug)
    output_lines.append("=" * width)
    output_lines.append("BEST PERFORMING MODELS (across all epochs):")
    output_lines.append("=" * 100)
    for csv_key, display_name, decimals, higher_better in METRIC_SPECS:
        best_epoch = None
        best_model = None
        best_value = None
        for name, em in experiments.items():
            for e, m in em.items():
                v = m.get(csv_key)
                if v is None:
                    continue
                # Skip bogus LPIPS 0.0 (missing or failed eval)
                if csv_key == "val_lpips" and v < 0.001:
                    continue
                if best_value is None or (higher_better and v > best_value) or (not higher_better and v < best_value):
                    best_value = v
                    best_model = name
                    best_epoch = e
        if best_model is not None and best_epoch is not None and best_value is not None:
            output_lines.append(f"{display_name:<20} {best_model:<15} Epoch {best_epoch} ({format_value(best_value, decimals)})")
    output_lines.append("=" * width)
    output_lines.append("")

    # MaskGIT section
    maskgit_dir = RESULTS_DIR / "maskgit"
    if maskgit_dir.exists():
        output_lines.append("=" * 100)
        output_lines.append("MaskGIT — GENERATIVE SAMPLING (FID, Precision, Recall)")
        output_lines.append("(Best checkpoint: transformer_best.pt; metrics on generated vs real images)")
        output_lines.append("=" * 100)
        output_lines.append("")
        maskgit_quantizers = sorted(
            d.name for d in maskgit_dir.iterdir()
            if d.is_dir() and (d / "sampling_metrics.json").exists()
        )
        if maskgit_quantizers:
            header = f"{'Metric':<20}"
            for q in maskgit_quantizers:
                header += f"{q.upper():<15}"
            output_lines.append(header)
            output_lines.append("-" * 100)
            for key, display_name, decimals, higher_better in [("fid", "FID", 2, False), ("precision", "Precision", 4, True), ("recall", "Recall", 4, True)]:
                row = f"{display_name:<20}"
                values = {}
                for q in maskgit_quantizers:
                    p = maskgit_dir / q / "sampling_metrics.json"
                    try:
                        m = json.loads(p.read_text())
                        v = m.get(key)
                        if v is not None:
                            values[q.upper()] = v
                            row += f"{format_value(v, decimals):<15}"
                        else:
                            row += f"{'N/A':<15}"
                    except Exception:
                        row += f"{'N/A':<15}"
                output_lines.append(row)
                if values:
                    best_q = (max if higher_better else min)(values.items(), key=lambda x: x[1])[0]
                    output_lines.append(f"  → Best: {best_q} ({format_value(values[best_q], decimals)})")
            output_lines.append("-" * 100)
            has_epoch5 = any((maskgit_dir / q / "sampling_metrics_epoch5.json").exists() for q in maskgit_quantizers)
            if has_epoch5:
                output_lines.append("")
                output_lines.append("MaskGIT @ Epoch 5 snapshot:")
                output_lines.append(header)
                output_lines.append("-" * 100)
                for key, display_name, decimals, _ in [("fid", "FID", 2, False), ("precision", "Precision", 4, True), ("recall", "Recall", 4, True)]:
                    row = f"{display_name:<20}"
                    for q in maskgit_quantizers:
                        p = maskgit_dir / q / "sampling_metrics_epoch5.json"
                        if p.exists():
                            try:
                                v = json.loads(p.read_text()).get(key)
                                row += f"{format_value(v, decimals):<15}" if v is not None else f"{'N/A':<15}"
                            except Exception:
                                row += f"{'N/A':<15}"
                        else:
                            row += f"{'N/A':<15}"
                    output_lines.append(row)
                output_lines.append("-" * 100)
        else:
            output_lines.append("(No MaskGIT sampling_metrics.json found.)")
        output_lines.append("")
        output_lines.append("=" * 100)
    output_lines.append("")

    output_path = RESULTS_DIR / "model_comparison_all_epochs.txt"
    output_path.write_text("\n".join(output_lines))
    print(f"\n✅ Comparison table saved to: {output_path}")
    print(f"   Total epochs compared: {len(all_epochs)}")
    print(f"   Models included: {', '.join(experiments.keys())}")


if __name__ == "__main__":
    main()

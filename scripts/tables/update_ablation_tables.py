#!/usr/bin/env python3
"""Regenerate ablation train/eval update tables from latest CSV rows."""

import csv
from pathlib import Path

RESULTS = Path("results/lmb")
ABLATIONS = Path("results/ablations")
RUNS = sorted([p.name for p in RESULTS.iterdir() if p.is_dir() and p.name.startswith("lmb_ablation_")])

def last_row(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return dict(rows[-1]) if rows else {}

def float_fmt(v):
    if v is None or v == "":
        return "N/A"
    try:
        x = float(v)
        return f"{x:.4f}" if abs(x) < 1e4 else f"{x:.2f}"
    except (ValueError, TypeError):
        return str(v)

def str_or_na(v):
    if v is None or v == "":
        return "N/A"
    return str(v)

def main():
    ABLATIONS.mkdir(parents=True, exist_ok=True)

    # Train table
    train_rows = []
    for run in RUNS:
        d = last_row(RESULTS / run / "train_metrics.csv")
        if not d:
            continue
        train_rows.append({
            "run": run,
            "epoch": str_or_na(d.get("epoch")),
            "step": str_or_na(d.get("global_step")),
            "rec_loss": float_fmt(d.get("rec_loss")),
            "vq_loss": float_fmt(d.get("vq_loss")),
            "total_loss": float_fmt(d.get("total_loss")),
            "perplexity": float_fmt(d.get("perplexity")),
            "active_codes": str_or_na(d.get("active_codes")),
        })
    with open(ABLATIONS / "lmb_ablation_updates_table.txt", "w") as f:
        f.write("run                                    epoch  step   rec_loss  vq_loss  total_loss  perplexity  active_codes\n")
        f.write("-------------------------------------  -----  -----  --------  -------  ----------  ----------  ------------\n")
        for r in train_rows:
            f.write(f"{r['run']:<37} {r['epoch']:>5}  {r['step']:>5}  {r['rec_loss']:>8}  {r['vq_loss']:>7}  {r['total_loss']:>10}  {r['perplexity']:>10}  {r['active_codes']:>12}\n")

    # Eval table
    eval_rows = []
    for run in RUNS:
        d = last_row(RESULTS / run / "eval_metrics.csv")
        if not d:
            continue
        eval_rows.append({
            "run": run,
            "epoch": str_or_na(d.get("epoch")),
            "rfid": float_fmt(d.get("val_rfid") or d.get("val_rFID") or d.get("rFID")),
            "psnr": float_fmt(d.get("val_psnr") or d.get("val_PSNR") or d.get("PSNR")),
            "ssim": float_fmt(d.get("val_ssim") or d.get("val_SSIM") or d.get("SSIM")),
            "lpips": float_fmt(d.get("val_lpips") or d.get("val_LPIPS") or d.get("LPIPS")),
            "rec_loss": float_fmt(d.get("val_rec_loss") or d.get("rec_loss")),
            "codebook_util": float_fmt(d.get("val_codebook_util") or d.get("codebook_util")),
            "perplexity": float_fmt(d.get("val_perplexity") or d.get("perplexity")),
            "active_codes": float_fmt(d.get("val_active_codes") or d.get("active_codes")),
        })
    with open(ABLATIONS / "lmb_ablation_eval_updates_table.txt", "w") as f:
        f.write("run                                    epoch  rfid      psnr     ssim    lpips   rec_loss  codebook_util  perplexity  active_codes\n")
        f.write("-------------------------------------  -----  --------  -------  ------  ------  --------  -------------  ----------  ------------\n")
        for r in eval_rows:
            f.write(f"{r['run']:<37} {r['epoch']:>5}  {r['rfid']:>8}  {r['psnr']:>7}  {r['ssim']:>6}  {r['lpips']:>6}  {r['rec_loss']:>8}  {r['codebook_util']:>13}  {r['perplexity']:>10}  {r['active_codes']:>12}\n")

    print("Updated", ABLATIONS / "lmb_ablation_updates_table.txt")
    print("Updated", ABLATIONS / "lmb_ablation_eval_updates_table.txt")

if __name__ == "__main__":
    main()

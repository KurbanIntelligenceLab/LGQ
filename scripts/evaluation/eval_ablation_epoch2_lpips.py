#!/usr/bin/env python3
"""Run evaluation (with LPIPS) on each LMB ablation run's epoch-2 checkpoint and update epoch-2 table."""

import re
import subprocess
import sys
from pathlib import Path

RESULTS_LMB = Path("results/lmb")
ABLATIONS = Path("results/ablations")
RUNS = [
    "lmb_ablation_reg_bins_only",
    "lmb_ablation_reg_none",
    "lmb_ablation_reg_peak_only",
    "lmb_ablation_reg_strong",
    "lmb_ablation_reg_weak",
]


def main():
    data_root = "data/imagenet"
    if not Path(data_root).exists() and Path("data/imagenet/val").exists():
        data_root = "data/imagenet"
    elif not Path(data_root).exists():
        # Try val in project root
        for d in ["data/imagenet", "data/imagenet/val", "data/val"]:
            if Path(d).exists():
                data_root = d
                break

    lpips_by_run = {}
    for run in RUNS:
        ckpt = RESULTS_LMB / run / "checkpoints" / "checkpoint_epoch_002.pt"
        if not ckpt.exists():
            print(f"Skip {run}: no epoch-2 checkpoint")
            lpips_by_run[run] = None
            continue
        cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint", str(ckpt),
            "--data-root", data_root,
            "--split", "test",
            "--no-fid",
            "--num-samples", "2000",
            "--batch-size", "32",
            "--gpu", "0",
        ]
        print(f"Running eval for {run}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=Path(__file__).resolve().parent.parent)
        out = result.stdout + result.stderr
        if result.returncode != 0:
            print(f"  Failed: {result.stderr[:500]}")
            lpips_by_run[run] = None
            continue
        # Parse "  lpips: 0.xxxx" (skip "nan")
        m = re.search(r"lpips:\s*([\d.]+)", out, re.I)
        if not m:
            m = re.search(r"'lpips':\s*([\d.]+)", out)
        if m:
            lpips_by_run[run] = float(m.group(1))
            print(f"  LPIPS = {lpips_by_run[run]:.4f}")
        else:
            lpips_by_run[run] = None
            print(f"  LPIPS not found in output")

    # Update epoch-2 table (read current rows from CSV epoch-2 data, add LPIPS)
    epoch2_data = {}
    for run in RUNS:
        csv_path = RESULTS_LMB / run / "eval_metrics.csv"
        if not csv_path.exists():
            continue
        with open(csv_path) as f:
            lines = f.readlines()
        header = lines[0].strip().split(",")
        for line in lines[1:]:
            row = line.strip().split(",")
            if len(row) < 2:
                continue
            try:
                ep = int(row[0])
            except ValueError:
                continue
            if ep != 2:
                continue
            d = dict(zip(header, row))
            d["run"] = run
            d["lpips_computed"] = lpips_by_run.get(run)
            epoch2_data[run] = d
            break

    # Build table lines (use existing CSV values; substitute LPIPS when we computed it)
    def num(s, fmt=".2f"):
        if s is None or s == "" or str(s).lower() in ("nan", "none"):
            return "—"
        try:
            return format(float(s), fmt)
        except (ValueError, TypeError):
            return str(s)

    rows = []
    for run in RUNS:
        d = epoch2_data.get(run)
        if not d:
            rows.append((run, None))
            continue
        rfid = num(d.get("val_rfid"), ".2f")
        psnr = num(d.get("val_psnr"), ".2f")
        ssim = num(d.get("val_ssim"), ".4f")
        lpips_val = lpips_by_run.get(run)
        if lpips_val is not None:
            lpips = num(lpips_val, ".4f")
        else:
            lpips = num(d.get("val_lpips"), ".4f")
        rec = num(d.get("val_rec_loss"), ".4f")
        util = num(d.get("val_codebook_util"), ".2f")
        perplexity = num(d.get("val_perplexity"), ".1f")
        active = num(d.get("val_active_codes"), ".0f")
        rows.append((run, (rfid, psnr, ssim, lpips, rec, util, perplexity, active)))

    out_path = ABLATIONS / "lmb_ablation_eval_epoch2.txt"
    ABLATIONS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("================================================================================\n")
        f.write("LMB REGULARIZATION ABLATION — EVAL AT EPOCH 2 (LPIPS computed)\n")
        f.write("================================================================================\n\n")
        f.write("Run                          Epoch   rFID     PSNR    SSIM    LPIPS   Rec Loss  Codebook Util  Perplexity  Active Codes\n")
        f.write("--------------------------------------------------------------------------------\n")
        for run, vals in rows:
            if vals is None:
                f.write(f"{run:<30}   —       —       —       —       —       —             —            —            —\n")
                continue
            rfid, psnr, ssim, lpips, rec, util, perplexity, active = vals
            util_str = util if util == "—" else f"{util}%"
            f.write(f"{run:<30}   2   {rfid:>7}  {psnr:>6}  {ssim:>6}  {lpips:>6}  {rec:>8}  {util_str:>12}  {perplexity:>10}  {active:>12}\n")
        f.write("--------------------------------------------------------------------------------\n\n")
    print(f"Updated {out_path}")

    # Also write LaTeX snippet with LPIPS (order: None, Weak, Bins only, Peak only, Strong)
    tex_path = ABLATIONS / "lmb_ablation_regularization_epoch2.tex"
    order = ["lmb_ablation_reg_none", "lmb_ablation_reg_weak", "lmb_ablation_reg_bins_only", "lmb_ablation_reg_peak_only", "lmb_ablation_reg_strong"]
    rows_dict = dict(rows)
    rows_ordered = [(r, rows_dict[r]) for r in order if r in rows_dict]

    rfids = []
    lpips_nums = []
    for run, vals in rows_ordered:
        if vals is None:
            continue
        rfid, psnr, ssim, lpips, rec, util, perplexity, active = vals
        if rfid != "—":
            try:
                rfids.append(float(rfid))
            except ValueError:
                pass
        if lpips != "—":
            try:
                lpips_nums.append(float(lpips))
            except ValueError:
                pass
    best_rfid = min(rfids) if rfids else None
    best_lpips = min(lpips_nums) if lpips_nums else None

    setting_names = {
        "lmb_ablation_reg_bins_only": "Bins only $(0,0.01)$",
        "lmb_ablation_reg_none": "None $(0,0)$",
        "lmb_ablation_reg_peak_only": "Peak only $(0.01,0)$",
        "lmb_ablation_reg_strong": "Strong $(0.01,0.01)$",
        "lmb_ablation_reg_weak": "Weak $(0.002,0.002)$",
    }
    latex_rows = []
    for run, vals in rows_ordered:
        if vals is None:
            continue
        rfid, psnr, ssim, lpips, rec, util, perplexity, active = vals
        util_tex = util if util == "—" else util
        lpips_tex = "---" if lpips == "—" else (f"\\textbf{{{lpips}}}" if best_lpips is not None and abs(float(lpips) - best_lpips) < 1e-6 else lpips)
        rfid_tex = f"\\textbf{{{rfid}}}" if best_rfid is not None and rfid != "—" and abs(float(rfid) - best_rfid) < 1e-6 else rfid
        setting = setting_names.get(run, run)
        latex_rows.append(f"{setting} &\n{rfid_tex} &\n{psnr} &\n{ssim} &\n{lpips_tex} &\n{util_tex} &\n{active} \\\\")
    latex_body = "\n\n".join(latex_rows)
    tex_content = f"""\\begin{{table}}[!h]
\\centering
\\caption{{\\textbf{{Effect of regularization on LGQ.}}
Ablation of peakedness and usage regularization.   \\\\
Metrics reported at epoch 2 (LPIPS computed).}}
\\label{{tab:lmb_ablation_regularization}}
\\begin{{tabular}}{{lcccccc}}
\\toprule
\\textbf{{Setting}} &
\\textbf{{rFID}} $\\downarrow$ &
\\textbf{{PSNR}} $\\uparrow$ &
\\textbf{{SSIM}} $\\uparrow$ &
\\textbf{{LPIPS}} $\\downarrow$ &
\\textbf{{Util. (\\%)}} $\\uparrow$ &
\\textbf{{Active}} \\\\
\\midrule
{latex_body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    with open(tex_path, "w") as f:
        f.write(tex_content)
    print(f"Updated {tex_path}")


if __name__ == "__main__":
    main()

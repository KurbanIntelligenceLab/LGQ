#!/usr/bin/env python3
"""Generate FSQ vs LMB per-channel comparison table (16K codebook)."""

import csv
from pathlib import Path

RESULTS = Path("results")
OUT_PATH = RESULTS / "fsq_lmb_perchannel_comparison.txt"

EXPERIMENTS = {
    "FSQ (16K)": RESULTS / "fsq/fsq_lv16-16-8-8_bs32_lr3e-4_dim128_20260123_184222_deab",
    "LMB Per-Ch Fair (16K)": RESULTS / "lmb/lmb_ablation_perchannel_fair",
    "LMB Flattened (16K)": RESULTS / "lmb/lmb_ablation_flattened",
}


def load_eval_metrics(exp_dir: Path):
    path = exp_dir / "eval_metrics.csv"
    if not path.exists():
        return None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    # De-duplicate by epoch: keep last row per epoch
    by_epoch = {}
    for r in rows:
        e = int(r.get("epoch", 0))
        by_epoch[e] = r
    return [by_epoch[e] for e in sorted(by_epoch.keys())]


def _get(r, key, fmt):
    v = r.get(key)
    if v is None or v == "" or str(v).lower() == "nan":
        return None
    try:
        x = float(v)
        return fmt(x)
    except (ValueError, TypeError):
        return None


def format_row(name, epoch, r):
    rfid = _get(r, "val_rfid", lambda x: f"{x:.2f}") or "N/A"
    psnr = _get(r, "val_psnr", lambda x: f"{x:.2f}") or "N/A"
    ssim = _get(r, "val_ssim", lambda x: f"{x:.4f}") or "N/A"
    lpips = _get(r, "val_lpips", lambda x: f"{x:.4f}") or "N/A"
    rec = _get(r, "val_rec_loss", lambda x: f"{x:.4f}") or "N/A"
    util = _get(r, "val_codebook_util", lambda x: f"{x:.2f}%") or "N/A"
    active = _get(r, "val_active_codes", lambda x: f"{int(x)}") or "N/A"
    return f"{name:<33} {epoch:<6} {rfid:<10} {psnr:<8} {ssim:<8} {lpips:<8} {rec:<10} {util:<15} {active:<15}"


def _latex_val(r, key, decimals):
    v = r.get(key)
    if v is None or v == "" or str(v).lower() == "nan":
        return "---"
    try:
        x = float(v)
        return f"{x:.{decimals}f}"
    except (ValueError, TypeError):
        return "---"


def _escape_latex(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def write_latex(data: dict, out_path: Path) -> None:
    """Write LaTeX table: epoch 3 comparison (FSQ, LMB Per-Ch Fair, LMB Flattened)."""
    epoch3_fsq = next((r for r in (data["FSQ (16K)"] or []) if int(r.get("epoch", 0)) == 3), None)
    epoch3_lmb = next((r for r in (data["LMB Per-Ch Fair (16K)"] or []) if int(r.get("epoch", 0)) == 3), None)
    epoch3_flat = next((r for r in (data["LMB Flattened (16K)"] or []) if int(r.get("epoch", 0)) == 3), None)

    rows = []
    for label, r in [
        ("FSQ (16K)", epoch3_fsq),
        ("LMB Per-Ch Fair (16K)", epoch3_lmb),
        ("LMB Flattened (16K)", epoch3_flat),
    ]:
        if r is None:
            rows.append((_escape_latex(label), "3", "---", "---", "---", "---", "---", "---", "---"))
            continue
        util_raw = _get(r, "val_codebook_util", lambda x: x)
        util = f"{util_raw:.2f}\\%" if util_raw is not None else "---"
        active = _get(r, "val_active_codes", lambda x: int(x))
        rows.append((
            _escape_latex(label),
            "3",
            _latex_val(r, "val_rfid", 2),
            _latex_val(r, "val_psnr", 2),
            _latex_val(r, "val_ssim", 4),
            _latex_val(r, "val_lpips", 4),
            _latex_val(r, "val_rec_loss", 4),
            util,
            f"{active}" if active is not None else "---",
        ))

    lines = [
        "% FSQ vs LMB Per-Channel Comparison (16K, Epoch 3)",
        "% Requires \\usepackage{booktabs}",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\begin{tabular}{llcccccccc}",
        "\\toprule",
        "Model & Epoch & rFID & PSNR & SSIM & LPIPS & Rec Loss & CB Util (\\%) & Active \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{FSQ vs LMB per-channel comparison (16K codebook) at epoch 3.}",
        "\\label{tab:fsq_lmb_perchannel_epoch3}",
        "\\end{table}",
    ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main():
    data = {}
    for label, exp_dir in EXPERIMENTS.items():
        rows = load_eval_metrics(exp_dir)
        data[label] = rows

    lines = []
    lines.append("=" * 120)
    lines.append("FSQ vs LMB PER-CHANNEL COMPARISON (16K Codebook Size)")
    lines.append("=" * 120)
    lines.append("")
    lines.append(
        f"{'Model':<33} {'Epoch':<6} {'rFID':<10} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8} {'Rec Loss':<10} {'Codebook Util':<15} {'Active Codes':<15}"
    )
    lines.append("-" * 120)

    # --- Epoch 3 comparison (fair: both at same epoch) ---
    epoch3_fsq = next((r for r in (data["FSQ (16K)"] or []) if int(r.get("epoch", 0)) == 3), None)
    epoch3_lmb = next((r for r in (data["LMB Per-Ch Fair (16K)"] or []) if int(r.get("epoch", 0)) == 3), None)
    if epoch3_fsq is not None or epoch3_lmb is not None:
        lines.append("Epoch 3 comparison (FSQ vs LMB Per-Ch Fair, same epoch):")
        lines.append("")
        if epoch3_fsq is not None:
            lines.append(format_row("FSQ (16K)", 3, epoch3_fsq))
        else:
            lines.append(format_row("FSQ (16K)", "N/A", {}))
        if epoch3_lmb is not None:
            lines.append(format_row("LMB Per-Ch Fair (16K)", 3, epoch3_lmb))
        else:
            lines.append(format_row("LMB Per-Ch Fair (16K)", "N/A", {}))
        lines.append("")
        lines.append("-" * 120)

    # FSQ: latest epoch only
    if data["FSQ (16K)"]:
        r = data["FSQ (16K)"][-1]
        ep = int(r.get("epoch", 0))
        lines.append(format_row("FSQ (16K)", ep, r))
    else:
        lines.append(format_row("FSQ (16K)", "N/A", {}))

    # LMB Per-Ch Fair: latest epoch only
    if data["LMB Per-Ch Fair (16K)"]:
        r = data["LMB Per-Ch Fair (16K)"][-1]
        ep = int(r.get("epoch", 0))
        lines.append(format_row("LMB Per-Ch Fair (16K)", ep, r))
    else:
        lines.append(format_row("LMB Per-Ch Fair (16K)", "N/A", {}))

    # LMB Flattened: all epochs (one row per epoch)
    if data["LMB Flattened (16K)"]:
        for i, r in enumerate(data["LMB Flattened (16K)"]):
            if i > 0:
                lines.append("")
            ep = int(r.get("epoch", 0))
            lines.append(format_row("LMB Flattened (16K)", ep, r))
    else:
        lines.append("")
        lines.append(format_row("LMB Flattened (16K)", "N/A", {}))

    lines.append("")
    lines.append("=" * 120)
    lines.append("NOTES:")
    lines.append("  - FSQ (16K): Fixed scalar quantization with levels [16, 16, 8, 8] = 16384 codes")
    lines.append(
        "  - LMB Per-Ch Fair (16K): Learnable bins with same structure as FSQ [16, 16, 8, 8] = 16384 codes (per-channel scalar)"
    )
    lines.append(
        "  - LMB Per-Ch Fair Flattened (16K): Project to 4 channels, then vector quantize [16384, 4] = 16384 codes (VQ over 4-dim vectors)"
    )
    lines.append(
        "  - LMB Flattened (16K): Vector quantize full 128-dim embedding [16384, 128] = 16384 codes (VQ over 128-dim vectors)"
    )
    lines.append("=" * 120)

    out = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(out, encoding="utf-8")
    print(out)
    print(f"Wrote {OUT_PATH}")

    tex_path = RESULTS / "fsq_lmb_perchannel_comparison.tex"
    write_latex(data, tex_path)


if __name__ == "__main__":
    main()

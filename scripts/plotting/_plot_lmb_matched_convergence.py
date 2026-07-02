"""One-off: plot convergence of the in-flight LMB matched-K sweep."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "results" / "lmb"
OUT_DIR = RUN_DIR / "_convergence_2026-04-26"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUNS = [
    ("K=4096",  "lmb_abl1_ddp4_cb4096_s1234"),
    ("K=8192",  "lmb_abl1_ddp4_cb8192_s1234"),
    ("K=16384", "lmb_abl1_ddp4_cb16384_s1234"),
    ("K=32768", "lmb_abl1_ddp4_cb32768_s1234"),
    ("K=65536", "lmb_abl1_ddp4_cb65536_s1234"),
]

METRICS = [
    ("val_psnr",          "PSNR (dB)",          False),
    ("val_lpips",         "LPIPS",              False),
    ("val_rec_loss",      "Recon loss (L1)",    False),
    ("val_codebook_util", "Codebook util (%)",  False),
]


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load(run_name: str):
    path = RUN_DIR / run_name / "eval_metrics.csv"
    with path.open() as f:
        return list(csv.DictReader(f))


def series(rows, key):
    xs, ys = [], []
    for r in rows:
        v = to_float(r.get(key))
        if v is None or v != v:  # skip NaN
            continue
        # cb16384 has a column-shift bug from ep13: ssim becomes tokens_per_image (256).
        if key == "val_ssim" and v > 1.5:
            continue
        xs.append(int(r["epoch"]))
        ys.append(v)
    return xs, ys


def main():
    data = {label: load(name) for label, name in RUNS}

    # Combined 2x2 panel.
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (key, ylabel, _) in zip(axes.flat, METRICS):
        for label, _ in RUNS:
            xs, ys = series(data[label], key)
            ax.plot(xs, ys, marker="o", label=label, linewidth=1.5, markersize=4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
    axes.flat[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("LMB matched-K sweep — convergence (as of 2026-04-26)", fontsize=13)
    out = OUT_DIR / "convergence_panel.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")

    # One PNG per metric (larger, easier to read individually).
    for key, ylabel, _ in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for label, _ in RUNS:
            xs, ys = series(data[label], key)
            ax.plot(xs, ys, marker="o", label=label, linewidth=1.6, markersize=5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs epoch — LMB matched-K (2026-04-26)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        slug = key.replace("val_", "")
        out = OUT_DIR / f"convergence_{slug}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

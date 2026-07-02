#!/usr/bin/env python3
"""
Center drift (adaptation) analysis: Δc = ‖c^(t) - c^(t-1)‖.

Frames drift as adaptation, not motion:
- FSQ: centers fixed → no drift (mismatch with data)
- VQ: centers drift only where activated → dead zones stay put
- LMB (LGQ): centers move early → align to data; stabilize later → consistent tokens

Outputs:
- Mean drift per epoch (mean over codes of Δc)
- Variance of drift across codes (per epoch transition)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def get_centers_from_state_dict(state_dict: dict, model_type: Optional[str] = None) -> Optional[torch.Tensor]:
    """
    Extract codebook centers from model state dict.
    Returns tensor with codes as rows: shape [num_codes, dim].
    For per-channel LMB, dim=1 (scalar per bin).
    """
    # LMB: quantize.centers
    for key, value in state_dict.items():
        if "quantize.centers" in key:
            c = value.detach().cpu()
            # Flattened: [K, C]; per-channel: [C, K]
            if c.dim() == 2 and c.shape[0] < c.shape[1]:
                # Likely [C, K] per-channel → treat each element as a 1D code
                c = c.flatten().unsqueeze(1)  # [C*K, 1]
            return c
    # VQ / SimVQ: quantize._codebook.embed  [num_codebooks, codebook_size, dim]
    for key, value in state_dict.items():
        if "quantize._codebook.embed" in key:
            c = value.detach().cpu()
            if c.dim() == 3:
                c = c.squeeze(0)  # [K, D]
            return c
    return None


def compute_drift_per_code(centers_prev: torch.Tensor, centers_curr: torch.Tensor) -> np.ndarray:
    """
    Δc = ‖c^(t) - c^(t-1)‖ per code (L2 norm per row).
    centers_prev, centers_curr: [num_codes, dim].
    Returns: shape (num_codes,) of per-code drifts.
    """
    if centers_prev.shape != centers_curr.shape:
        return np.array([])
    diff = centers_curr - centers_prev  # [N, D]
    # L2 norm per row
    if diff.shape[1] == 1:
        drift = diff.abs().squeeze(1).numpy()
    else:
        drift = torch.norm(diff, p=2, dim=1).numpy()
    return drift


def compute_drift_stats_for_experiment(
    exp_dir: Path,
    checkpoint_subdir: str = "checkpoints",
    model_type_hint: Optional[str] = None,
) -> Optional[dict]:
    """
    Load consecutive checkpoints, compute mean drift and variance across codes per epoch.
    Returns dict with epochs, mean_drift, var_drift, and model_type; or None if no centers/checkpoints.
    """
    ckpt_dir = exp_dir / checkpoint_subdir
    if not ckpt_dir.exists():
        return None

    ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if len(ckpts) < 2:
        return None

    # Optionally restrict by config model type
    config_path = exp_dir / "config.json"
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    model_type = model_type_hint or config.get("model", "")

    # FSQ has no learnable codebook
    if model_type == "fsq":
        return {
            "model_type": "fsq",
            "epochs": [],
            "mean_drift": [],
            "var_drift": [],
            "num_codes": 0,
            "message": "FSQ has fixed centers (no drift).",
        }

    epoch_centers: dict[int, torch.Tensor] = {}
    for p in ckpts:
        try:
            epoch = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        except Exception:
            continue
        state = ckpt.get("model_state_dict", {})
        centers = get_centers_from_state_dict(state, model_type)
        if centers is not None:
            epoch_centers[epoch] = centers

    if len(epoch_centers) < 2:
        return None

    epochs_sorted = sorted(epoch_centers.keys())
    mean_drifts = []
    var_drifts = []
    transition_epochs = []  # epoch we're transitioning TO (t)

    for i in range(len(epochs_sorted) - 1):
        e_prev, e_curr = epochs_sorted[i], epochs_sorted[i + 1]
        c_prev = epoch_centers[e_prev]
        c_curr = epoch_centers[e_curr]
        if c_prev.shape != c_curr.shape:
            continue
        drift = compute_drift_per_code(c_prev, c_curr)
        if drift.size == 0:
            continue
        mean_drifts.append(float(np.mean(drift)))
        var_drifts.append(float(np.var(drift)))
        transition_epochs.append(e_curr)

    if not mean_drifts:
        return None

    num_codes = epoch_centers[epochs_sorted[0]].shape[0]
    return {
        "model_type": model_type,
        "exp_name": exp_dir.name,
        "epochs": transition_epochs,
        "mean_drift": mean_drifts,
        "var_drift": var_drifts,
        "num_codes": num_codes,
        "epoch_range": (min(epochs_sorted), max(epochs_sorted)),
    }


def plot_drift(
    results: list[dict],
    output_path: Path,
    title: str = "Center adaptation (drift) over training",
) -> None:
    """
    Plot mean drift per epoch and variance across codes (2 panels).
    Frames drift as adaptation: centers move early to align to data, stabilize later.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    for r in results:
        if not r.get("mean_drift"):
            continue
        epochs = r["epochs"]
        mean_d = r["mean_drift"]
        var_d = r["var_drift"]
        label = f"{r.get('model_type', '?')} ({r.get('exp_name', '')})"
        ax1.plot(epochs, mean_d, "o-", label=label, markersize=4)
        ax2.plot(epochs, var_d, "o-", label=label, markersize=4)

    ax1.set_ylabel(r"Mean drift $\bar{\Delta}_c$")
    ax1.set_title(r"Mean drift per epoch — $\Delta_c = \| c^{(t)} - c^{(t-1)} \|$ (adaptation)")
    ax1.legend(loc="best", fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"Variance of $\Delta_c$ across codes")
    ax2.set_title("Variance of drift across codes")
    ax2.legend(loc="best", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze center drift (Δc) between consecutive checkpoints; plot mean and variance."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Experiment directories (each with checkpoints/). If empty, use defaults.",
    )
    parser.add_argument("--output", "-o", type=Path, default=Path("results/plots/center_drift.png"))
    parser.add_argument("--csv", type=Path, default=None, help="Save per-experiment drift stats to CSV")
    parser.add_argument("--no-plot", action="store_true", help="Only compute stats, do not plot")
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    args = parser.parse_args()

    if args.experiment_dirs:
        exp_dirs = [Path(d) for d in args.experiment_dirs]
    else:
        # Default: one LMB and one VQ run if available
        exp_dirs = []
        for sub in ("lmb", "vq", "sim_vq"):
            parent = args.results_root / sub
            if parent.exists():
                # Prefer runs that have checkpoints
                for d in sorted(parent.iterdir()):
                    if d.is_dir() and (d / "checkpoints").exists():
                        exp_dirs.append(d)
                        break
        if not exp_dirs:
            # Fallback: any dir under results with checkpoints
            for d in args.results_root.rglob("checkpoints"):
                exp_dirs.append(d.parent)
            exp_dirs = list(dict.fromkeys(exp_dirs))[:6]

    results = []
    for exp_dir in exp_dirs:
        if not exp_dir.exists():
            print(f"  Skip (not found): {exp_dir}")
            continue
        r = compute_drift_stats_for_experiment(exp_dir)
        if r is None:
            print(f"  Skip (no centers or <2 ckpts): {exp_dir}")
            continue
        results.append(r)
        print(f"  {r.get('exp_name', exp_dir.name)} ({r.get('model_type')}): "
              f"epochs {r['epoch_range']}, mean_drift last={r['mean_drift'][-1]:.6f}, var_drift last={r['var_drift'][-1]:.6f}")

    if not results:
        print("No experiments with center drift data. Ensure checkpoints exist and contain quantize.centers or quantize._codebook.embed.")
        return

    if args.csv:
        import csv
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["exp_name", "model_type", "epoch", "mean_drift", "var_drift", "num_codes"])
            for r in results:
                for i, ep in enumerate(r["epochs"]):
                    w.writerow([
                        r.get("exp_name", ""),
                        r.get("model_type", ""),
                        ep,
                        r["mean_drift"][i],
                        r["var_drift"][i],
                        r.get("num_codes", ""),
                    ])
        print(f"  Saved CSV: {args.csv}")

    if not args.no_plot:
        plot_drift(results, args.output, title="Center adaptation (drift) over training")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Evaluate rFID for last-epoch checkpoints of all models using pytorch-fid library.

Finds the last-epoch checkpoint per model, runs evaluation with rFID (now using
pytorch-fid), and prints a comparison table.

Usage:
    python scripts/evaluate_rfid_last_epochs.py --data-root ~/data/imagenet
    python scripts/evaluate_rfid_last_epochs.py --data-root ~/data/imagenet --fid-samples 5000 --write-csv
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

RESULTS_DIR = Path("results")
MODEL_NAMES = ["FSQ", "LFQ", "LMB", "LMB-Fixed", "SIM_VQ", "VQ"]


def find_experiment(model_name: str, image_size: int = 128) -> Path | None:
    """Find the experiment directory for a model (128x128, cb 16384 preferred)."""
    if model_name == "FSQ":
        candidates_16384 = []
        candidates_other = []
        for exp_dir in (RESULTS_DIR / "fsq").iterdir():
            if not exp_dir.is_dir():
                continue
            config_path = exp_dir / "config.json"
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
                eval_path = exp_dir / "eval_metrics.csv"
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
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                eval_path = exp_dir / "eval_metrics.csv"
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
            if not config_path.exists():
                continue
            try:
                with open(config_path) as f:
                    config = json.load(f)
                if config.get("image_size") != image_size:
                    continue
                eval_path = exp_dir / "eval_metrics.csv"
                n = len(list(csv.DictReader(open(eval_path)))) if eval_path.exists() else 0
                candidates.append((exp_dir, n))
            except Exception:
                pass
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
    return None


def get_last_epoch_checkpoint(exp_dir: Path) -> Path | None:
    """Return the checkpoint with highest epoch number, or best_model.pt / latest_model.pt."""
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    epoch_files = list(ckpt_dir.glob("checkpoint_epoch_*.pt"))
    if epoch_files:
        last_epoch = -1
        chosen = None
        for f in epoch_files:
            try:
                epoch = int(f.stem.rsplit("_", 1)[-1])
                if epoch > last_epoch:
                    last_epoch = epoch
                    chosen = f
            except (ValueError, IndexError):
                continue
        return chosen
    # Fallback to best or latest
    for name in ("latest_model.pt", "best_model.pt"):
        p = ckpt_dir / name
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate rFID for last-epoch checkpoints (using pytorch-fid)"
    )
    ap.add_argument("--data-root", type=str, default="data/imagenet", help="ImageNet data root")
    ap.add_argument("--split", type=str, default=None, help="Split: val, test, or train (default: val if exists else test)")
    ap.add_argument("--fid-samples", type=int, default=10000, help="Samples for rFID")
    ap.add_argument("--gpu", type=int, default=0, help="GPU ID")
    ap.add_argument("--write-csv", action="store_true", help="Update eval_metrics.csv per experiment")
    ap.add_argument("--num-samples", type=int, default=None, help="Max total samples (default: all)")
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    split = args.split
    if split is None:
        split = "val" if os.path.exists(os.path.join(data_root, "val")) else "test"
    data_path = os.path.join(data_root, split)
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Set --data-root and/or --split.")
        sys.exit(1)

    print("rFID evaluation using pytorch-fid library")
    print("=" * 70)

    results = []
    for model_name in MODEL_NAMES:
        exp_dir = find_experiment(model_name)
        if exp_dir is None:
            print(f"  {model_name}: no experiment found")
            results.append((model_name, None, None, "N/A"))
            continue

        ckpt = get_last_epoch_checkpoint(exp_dir)
        if ckpt is None:
            print(f"  {model_name}: no checkpoint found in {exp_dir}")
            results.append((model_name, exp_dir, None, "N/A"))
            continue

        epoch = "?"
        if "checkpoint_epoch_" in ckpt.name:
            try:
                epoch = ckpt.stem.rsplit("_", 1)[-1]
            except Exception:
                pass

        cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint", str(ckpt),
            "--data-root", data_root,
            "--split", split,
            "--fid-samples", str(args.fid_samples),
            "--gpu", str(args.gpu),
        ]
        if args.num_samples:
            cmd.extend(["--num-samples", str(args.num_samples)])
        if args.write_csv:
            cmd.extend(["--write-eval-csv", str(exp_dir)])

        print(f"  {model_name}: evaluating {ckpt.name} (epoch {epoch})...")
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent)
        if r.returncode != 0:
            print(f"    Error: {r.stderr[:500] if r.stderr else r.stdout[:500]}")
            results.append((model_name, exp_dir, epoch, "FAIL"))
            continue

        # Parse rfid from output (e.g. "  rfid: 110.51")
        rfid_val = "N/A"
        for line in r.stdout.splitlines():
            if "rfid:" in line.lower():
                try:
                    rfid_val = line.split(":")[-1].strip()
                    break
                except Exception:
                    pass

        results.append((model_name, exp_dir, epoch, rfid_val))
        print(f"    rFID = {rfid_val}")

    print("\n" + "=" * 70)
    print("rFID (pytorch-fid) — last epoch per model")
    print("=" * 70)
    print(f"{'Model':<15} {'Epoch':<8} {'rFID':<12}")
    print("-" * 40)
    for name, _, epoch, rfid in results:
        print(f"{name:<15} {str(epoch):<8} {str(rfid):<12}")
    print("=" * 70)


if __name__ == "__main__":
    main()

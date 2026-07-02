#!/usr/bin/env python3
"""
Find latest ROT_VQ 128 run with a checkpoint and run evaluation, writing eval_metrics.csv
so the model comparison table can include ROT_VQ.

Usage:
    python scripts/evaluate_rot_vq.py
    python scripts/evaluate_rot_vq.py --data-root ~/data/imagenet --gpu 0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Evaluate ROT_VQ and write eval_metrics.csv")
    parser.add_argument("--results-dir", type=str, default="results", help="Results root")
    parser.add_argument("--data-root", type=str, default="~/data/imagenet", help="ImageNet root")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fid-samples", type=int, default=2000, help="Samples for rFID")
    parser.add_argument("--num-samples", type=int, default=None, help="Max eval samples (default: all)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rot_vq_dir = results_dir / "rot_vq"
    if not rot_vq_dir.exists():
        print("No results/rot_vq directory found.")
        sys.exit(1)

    # Find run with image_size 128 and checkpoints/latest_model.pt
    best_run = None
    for exp_dir in sorted(rot_vq_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not exp_dir.is_dir():
            continue
        config_path = exp_dir / "config.json"
        ckpt_path = exp_dir / "checkpoints" / "latest_model.pt"
        if not config_path.exists() or not ckpt_path.exists():
            continue
        try:
            with open(config_path) as f:
                config = json.load(f)
            if config.get("image_size") != 128:
                continue
            best_run = exp_dir
            break
        except Exception:
            continue

    if best_run is None:
        print("No ROT_VQ run with image_size=128 and checkpoints/latest_model.pt found.")
        print("Train at least 1 epoch: bash shell/train_rot_vq.sh --image-size 128 --epochs 1 --gpu 0")
        sys.exit(1)

    ckpt = best_run / "checkpoints" / "latest_model.pt"
    cmd = [
        sys.executable,
        "scripts/evaluate.py",
        "--checkpoint", str(ckpt),
        "--write-eval-csv", str(best_run),
        "--data-root", str(Path(args.data_root).expanduser()),
        "--gpu", str(args.gpu),
        "--fid-samples", str(args.fid_samples),
    ]
    if args.num_samples is not None:
        cmd += ["--num-samples", str(args.num_samples)]

    print(f"Evaluating ROT_VQ: {ckpt}")
    print(f"Run dir: {best_run}")
    ret = subprocess.run(cmd, cwd=Path(__file__).resolve().parent.parent)
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()

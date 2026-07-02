#!/usr/bin/env python3
"""
Re-compute rFID for specific epochs using pytorch-fid and more samples (e.g. 10k).

Updates eval_metrics.csv in place with the new val_rfid values.

Usage:
    python scripts/recompute_rfid_epochs.py --exp-dir results/lmb/lmb_nb16384_... --epochs 44,45,46 --fid-samples 10000
    python scripts/recompute_rfid_epochs.py --model LMB --epochs 44,45 --fid-samples 10000
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

TORCHVISION_AVAILABLE = False
try:
    from torchvision import datasets, transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    pass

RESULTS_DIR = Path("results")


def find_experiment(model_name: str, image_size: int = 128) -> Path | None:
    """Find experiment directory for a model."""
    if model_name == "LMB":
        for exp_dir in (RESULTS_DIR / "lmb").iterdir():
            if not exp_dir.is_dir() or exp_dir.name == "lmb_fixed_init":
                continue
            cfg = exp_dir / "config.json"
            if cfg.exists() and (exp_dir / "eval_metrics.csv").exists():
                with open(cfg) as f:
                    if json.load(f).get("image_size") == image_size:
                        return exp_dir
    elif model_name == "LMB-Fixed":
        exp_dir = RESULTS_DIR / "lmb" / "lmb_fixed_init"
        if exp_dir.exists() and (exp_dir / "config.json").exists():
            with open(exp_dir / "config.json") as f:
                if json.load(f).get("image_size") == image_size:
                    return exp_dir
    # Fallback: iterate common model dirs
    for subdir in ("fsq", "lfq", "lmb", "sim_vq", "vq"):
        for exp_dir in (RESULTS_DIR / subdir).iterdir():
            if not exp_dir.is_dir():
                continue
            cfg = exp_dir / "config.json"
            if cfg.exists() and (exp_dir / "eval_metrics.csv").exists():
                with open(cfg) as f:
                    if json.load(f).get("image_size") == image_size:
                        return exp_dir
    return None


def main():
    ap = argparse.ArgumentParser(
        description="Re-compute rFID for epochs using pytorch-fid and more samples"
    )
    ap.add_argument("--exp-dir", type=str, default=None, help="Experiment directory")
    ap.add_argument("--model", type=str, default=None, help="Model name (LMB, LMB-Fixed, etc.) to find exp-dir")
    ap.add_argument("--epochs", type=str, default=None, help="Comma-separated epochs (e.g. 44,45,46). Default: all with checkpoints")
    ap.add_argument("--fid-samples", type=int, default=10000, help="Number of samples for rFID")
    ap.add_argument("--data-root", type=str, default="data/imagenet")
    ap.add_argument("--split", type=str, default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done, don't run")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir) if args.exp_dir else None
    if exp_dir is None and args.model:
        exp_dir = find_experiment(args.model)
    if exp_dir is None or not exp_dir.exists():
        print(f"Error: need --exp-dir or --model. Exp dir: {exp_dir}")
        sys.exit(1)

    data_root = os.path.expanduser(args.data_root)
    split = args.split or ("val" if os.path.exists(os.path.join(data_root, "val")) else "test")
    data_path = os.path.join(data_root, split)
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found")
        sys.exit(1)

    # Collect epochs to process
    ckpt_dir = exp_dir / "checkpoints"
    if not ckpt_dir.exists():
        print(f"Error: {ckpt_dir} not found")
        sys.exit(1)

    all_epochs = set()
    for f in ckpt_dir.iterdir():
        if f.suffix == ".pt" and "checkpoint_epoch_" in f.name:
            try:
                ep = int(f.stem.rsplit("_", 1)[-1])
                all_epochs.add(ep)
            except ValueError:
                pass

    if args.epochs:
        target_epochs = sorted({int(x.strip()) for x in args.epochs.split(",")})
        target_epochs = [e for e in target_epochs if e in all_epochs]
    else:
        target_epochs = sorted(all_epochs)

    if not target_epochs:
        print("No epochs to process")
        sys.exit(0)

    print(f"Exp dir: {exp_dir}")
    print(f"Epochs: {target_epochs}")
    print(f"rFID samples: {args.fid_samples}")
    print(f"Data: {data_path}")
    if args.dry_run:
        print("(dry run, exiting)")
        sys.exit(0)

    # Lazy imports (after arg parsing, in case of --dry-run)
    from scripts.evaluation.evaluate import load_model_from_checkpoint, evaluate
    from scripts.evaluation.fid import FIDCalculator

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    image_size = 128
    try:
        with open(exp_dir / "config.json") as f:
            image_size = int(json.load(f).get("image_size", 128))
    except Exception:
        pass

    if not TORCHVISION_AVAILABLE:
        print("Error: torchvision required for dataset loading")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=data_path, transform=transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    fid_calculator = FIDCalculator(device=device)
    # Reset backend log so we see which backend is used
    if hasattr(fid_calculator, "_backend_logged"):
        fid_calculator._backend_logged = False
    csv_path = exp_dir / "eval_metrics.csv"

    # Read existing CSV
    rows_by_epoch = {}
    fieldnames = []
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                try:
                    ep = int(row.get("epoch", 0))
                    rows_by_epoch[ep] = row
                except (ValueError, TypeError):
                    pass

    default_fieldnames = [
        "epoch", "global_step", "val_active_codes", "val_codebook_util", "val_lpips",
        "val_perplexity", "val_psnr", "val_rec_loss", "val_rfid", "val_ssim", "val_total_loss",
        "val_entropy_bits", "val_bits_per_token",
    ]
    if not fieldnames:
        fieldnames = default_fieldnames

    for epoch in target_epochs:
        ckpt = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
        if not ckpt.exists():
            print(f"  Epoch {epoch}: checkpoint not found, skip")
            continue

        print(f"  Epoch {epoch}: loading {ckpt.name}...")
        model, model_config, train_args, checkpoint = load_model_from_checkpoint(str(ckpt), device)

        print(f"    Computing rFID with {args.fid_samples} samples (pytorch-fid)...")
        codebook_size = None
        if hasattr(model, "quantize") and hasattr(model.quantize, "K"):
            codebook_size = model.quantize.K
        metrics = evaluate(
            model, model_config, loader, device,
            max_samples=None, compute_fid=True, fid_samples=args.fid_samples,
            fid_calculator=fid_calculator, codebook_size=codebook_size,
        )
        new_rfid = metrics.get("rfid", float("nan"))

        if epoch in rows_by_epoch:
            rows_by_epoch[epoch]["val_rfid"] = new_rfid
            print(f"    Updated val_rfid = {new_rfid:.2f}")
        else:
            print(f"    Warning: epoch {epoch} not in eval_metrics.csv, skipping (use evaluate.py --write-eval-csv to add)")

    # Write back CSV (all epochs, sorted)
    all_epochs_sorted = sorted(rows_by_epoch.keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for ep in all_epochs_sorted:
            writer.writerow(rows_by_epoch[ep])

    print(f"Updated {csv_path}")


if __name__ == "__main__":
    main()

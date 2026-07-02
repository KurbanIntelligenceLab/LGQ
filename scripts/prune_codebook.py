#!/usr/bin/env python3
"""
Post-training codebook pruning for LMB quantizers.

Permanently removes unused/underused codebook entries from a trained
MultiBinDiscretizer checkpoint, producing a smaller model.

Usage:
    # Remove codes with usage < 0.01% (default)
    python scripts/prune_codebook.py \
        --checkpoint results/lmb/lmb_cb16384_256_s1234/checkpoints/best_model.pt \
        --val-dir Data/imagenet/val

    # Keep only the top 4096 most-used codes
    python scripts/prune_codebook.py \
        --checkpoint results/lmb/.../best_model.pt \
        --val-dir Data/imagenet/val \
        --strategy keep-k --keep-k 4096

    # Remove bottom 30% by usage
    python scripts/prune_codebook.py \
        --checkpoint results/lmb/.../best_model.pt \
        --val-dir Data/imagenet/val \
        --strategy fraction --prune-fraction 0.3
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quantization.quantizers import (
    MultiBinDiscretizer,
    MultiBinDiscretizerFair,
    prune_codebook,
)
from scripts.evaluation.evaluate import load_model_from_checkpoint, evaluate, FlatImageDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-training codebook pruning for LMB quantizers"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained LMB checkpoint"
    )
    parser.add_argument(
        "--val-dir", type=str, required=True, help="Path to validation image directory"
    )

    # Pruning strategy
    parser.add_argument(
        "--strategy",
        type=str,
        default="threshold",
        choices=["threshold", "fraction", "keep-k"],
        help="Pruning strategy (default: threshold)",
    )
    parser.add_argument(
        "--usage-threshold",
        type=float,
        default=1e-4,
        help="Remove codes with usage fraction below this (strategy=threshold)",
    )
    parser.add_argument(
        "--prune-fraction",
        type=float,
        default=0.2,
        help="Fraction of codes to remove (strategy=fraction)",
    )
    parser.add_argument(
        "--keep-k", type=int, default=None, help="Keep top-K codes (strategy=keep-k)"
    )

    # Evaluation
    parser.add_argument(
        "--eval-before",
        action="store_true",
        default=True,
        help="Evaluate before pruning",
    )
    parser.add_argument(
        "--no-eval-before", action="store_true", help="Skip pre-pruning evaluation"
    )
    parser.add_argument(
        "--eval-after",
        action="store_true",
        default=True,
        help="Evaluate after pruning",
    )
    parser.add_argument(
        "--no-eval-after", action="store_true", help="Skip post-pruning evaluation"
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Limit evaluation samples",
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for pruned checkpoint (default: <input>_pruned.pt)",
    )
    parser.add_argument(
        "--save-usage",
        type=str,
        default=None,
        help="Save usage histogram to .pt file",
    )

    # Data loading
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)

    return parser.parse_args()


def make_val_loader(val_dir: str, image_size: int, batch_size: int, num_workers: int):
    """Create validation data loader."""
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    try:
        val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    except (FileNotFoundError, RuntimeError):
        val_dataset = FlatImageDataset(root=val_dir, transform=transform)

    return DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )


@torch.no_grad()
def compute_usage(model, val_loader, device) -> torch.Tensor:
    """
    Compute hard-assignment usage histogram over the validation set.

    Returns:
        For MultiBinDiscretizer (flattened): LongTensor [K] with counts.
        For MultiBinDiscretizer (per-channel): LongTensor [C, K] with counts.
        For MultiBinDiscretizerFair: list of C LongTensors, each [levels[c]].
    """
    model.eval()
    quantizer = model.quantize
    is_fair = isinstance(quantizer, MultiBinDiscretizerFair)
    is_flat = not is_fair and getattr(quantizer, "flatten_channels", False)

    if is_fair:
        counts = [
            torch.zeros(quantizer.levels[c], dtype=torch.long, device=device)
            for c in range(quantizer.C)
        ]
    elif is_flat:
        counts = torch.zeros(quantizer.K, dtype=torch.long, device=device)
    else:
        counts = torch.zeros(quantizer.C, quantizer.K, dtype=torch.long, device=device)

    total_tokens = 0
    for x, _ in tqdm(val_loader, desc="Computing usage"):
        x = x.to(device)
        outputs = model(x)
        indices = outputs[1]  # [B, T] or [B, T, C]

        if is_fair:
            # indices: [B, T, C]
            idx = indices.view(-1, quantizer.C)  # [N, C]
            for c in range(quantizer.C):
                counts[c] += torch.bincount(
                    idx[:, c], minlength=quantizer.levels[c]
                )
            total_tokens += idx.shape[0]
        elif is_flat:
            # indices: [B, T]
            idx = indices.view(-1)
            counts += torch.bincount(idx, minlength=quantizer.K)
            total_tokens += idx.shape[0]
        else:
            # indices: [B, T, C]
            idx = indices.view(-1, quantizer.C)  # [N, C]
            for c in range(quantizer.C):
                counts[c] += torch.bincount(
                    idx[:, c], minlength=quantizer.K
                )
            total_tokens += idx.shape[0]

    print(f"  Processed {total_tokens:,} tokens")
    return counts


def compute_keep_mask(counts, strategy, args, quantizer):
    """
    Determine which codes to keep based on the pruning strategy.

    Returns keep_mask in the same format as counts.
    """
    is_fair = isinstance(quantizer, MultiBinDiscretizerFair)

    if is_fair:
        keep_masks = []
        for c in range(quantizer.C):
            mask = _compute_mask_1d(
                counts[c], strategy, args, label=f"ch{c}"
            )
            keep_masks.append(mask)
        return keep_masks

    if counts.dim() == 1:
        # Flattened mode: [K]
        return _compute_mask_1d(counts, strategy, args)
    else:
        # Per-channel mode: [C, K] — conservative: keep if used in ANY channel
        total = counts.sum(dim=0)  # [K]
        return _compute_mask_1d(total, strategy, args)


def _compute_mask_1d(counts, strategy, args, label=""):
    """Compute a 1-D boolean keep mask from a counts vector."""
    K = len(counts)
    total = counts.sum().float()
    usage_frac = counts.float() / total.clamp(min=1)

    if strategy == "threshold":
        mask = usage_frac >= args.usage_threshold
    elif strategy == "fraction":
        n_remove = int(K * args.prune_fraction)
        _, sorted_idx = counts.sort()
        remove_set = set(sorted_idx[:n_remove].tolist())
        mask = torch.tensor(
            [i not in remove_set for i in range(K)],
            dtype=torch.bool,
            device=counts.device,
        )
    elif strategy == "keep-k":
        keep_k = args.keep_k
        assert keep_k is not None and keep_k > 0, "--keep-k must be a positive integer"
        keep_k = min(keep_k, K)
        _, sorted_idx = counts.sort(descending=True)
        keep_set = set(sorted_idx[:keep_k].tolist())
        mask = torch.tensor(
            [i in keep_set for i in range(K)],
            dtype=torch.bool,
            device=counts.device,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    kept = int(mask.sum().item())
    pfx = f"  [{label}] " if label else "  "
    print(f"{pfx}{kept}/{K} codes kept ({kept/K*100:.1f}%)")
    return mask


def print_comparison(before: Dict, after: Dict):
    """Print a before/after metrics comparison table."""
    keys = [
        ("Codebook size", "active_codes", "{:.0f}"),
        ("PSNR (dB)", "val_psnr", "{:.2f}"),
        ("SSIM", "val_ssim", "{:.4f}"),
        ("LPIPS", "val_lpips", "{:.4f}"),
        ("rFID", "val_rfid", "{:.2f}"),
        ("Perplexity", "val_perplexity", "{:.1f}"),
        ("Codebook util %", "val_codebook_util", "{:.1f}"),
    ]

    print(f"\n{'Metric':<20s} | {'Before':>10s} | {'After':>10s} | {'Delta':>10s}")
    print("-" * 60)
    for name, key, fmt in keys:
        b = before.get(key)
        a = after.get(key)
        if b is None or a is None:
            continue
        try:
            delta = a - b
            sign = "+" if delta >= 0 else ""
            print(
                f"{name:<20s} | {fmt.format(b):>10s} | {fmt.format(a):>10s} | {sign}{fmt.format(delta):>9s}"
            )
        except (TypeError, ValueError):
            pass


def main():
    args = parse_args()
    eval_before = args.eval_before and not args.no_eval_before
    eval_after = args.eval_after and not args.no_eval_after

    # Device
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu}"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, model_config_cls, model_args, checkpoint = load_model_from_checkpoint(
        args.checkpoint, device
    )
    config = checkpoint["config"]

    # Verify LMB
    quantizer = model.quantize
    if not isinstance(quantizer, (MultiBinDiscretizer, MultiBinDiscretizerFair)):
        print(f"ERROR: Pruning only supports LMB quantizers, got {type(quantizer).__name__}")
        sys.exit(1)

    is_fair = isinstance(quantizer, MultiBinDiscretizerFair)
    is_flat = not is_fair and getattr(quantizer, "flatten_channels", False)

    if is_fair:
        K_str = f"levels={quantizer.levels}, K_max={quantizer.K_max}"
    else:
        K_str = f"K={quantizer.K}, C={quantizer.C}"
    mode = "fair" if is_fair else ("flattened" if is_flat else "per-channel")
    print(f"  Quantizer: MultiBinDiscretizer{'Fair' if is_fair else ''} ({mode})")
    print(f"  {K_str}")

    # Data loader
    val_loader = make_val_loader(
        args.val_dir, args.image_size, args.batch_size, args.num_workers
    )
    print(f"  Val images: {len(val_loader.dataset):,}")

    # Evaluate before pruning
    before_metrics = {}
    if eval_before:
        print("\n--- Evaluating BEFORE pruning ---")
        before_metrics = evaluate(
            model,
            model_config_cls,
            val_loader,
            device,
            max_samples=args.max_eval_samples,
        )
        psnr = before_metrics.get("val_psnr", float("nan"))
        ssim = before_metrics.get("val_ssim", float("nan"))
        lpips = before_metrics.get("val_lpips", float("nan"))
        print(f"  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  LPIPS={lpips:.4f}")

    # Compute usage
    print("\nComputing codebook usage over validation set...")
    counts = compute_usage(model, val_loader, device)

    if args.save_usage:
        torch.save(counts, args.save_usage)
        print(f"  Saved usage histogram to {args.save_usage}")

    # Print usage summary
    if is_fair:
        for c in range(quantizer.C):
            active = int((counts[c] > 0).sum().item())
            print(f"  Channel {c}: {active}/{quantizer.levels[c]} active codes")
    elif counts.dim() == 1:
        active = int((counts > 0).sum().item())
        total_k = quantizer.K
        print(f"  Active codes: {active}/{total_k} ({active/total_k*100:.1f}%)")
    else:
        total_usage = counts.sum(dim=0)
        active = int((total_usage > 0).sum().item())
        total_k = quantizer.K
        print(f"  Active codes (any channel): {active}/{total_k} ({active/total_k*100:.1f}%)")

    # Compute keep mask
    strategy_desc = args.strategy
    if args.strategy == "threshold":
        strategy_desc += f" (usage < {args.usage_threshold})"
    elif args.strategy == "fraction":
        strategy_desc += f" (remove bottom {args.prune_fraction*100:.0f}%)"
    elif args.strategy == "keep-k":
        strategy_desc += f" (keep top {args.keep_k})"
    print(f"\nStrategy: {strategy_desc}")

    keep_mask = compute_keep_mask(counts, args.strategy, args, quantizer)

    # Record original size
    if is_fair:
        original_levels = list(quantizer.levels)
        original_codebook = int(math.prod(original_levels))
    else:
        original_K = quantizer.K

    # Prune
    print("\nPruning codebook...")
    old_to_new = prune_codebook(quantizer, keep_mask)

    if is_fair:
        new_codebook = int(math.prod(quantizer.levels))
        print(f"  levels: {original_levels} -> {quantizer.levels}")
        print(f"  Effective codebook: {original_codebook:,} -> {new_codebook:,}")
    else:
        print(f"  K: {original_K} -> {quantizer.K}")

    # Evaluate after pruning
    after_metrics = {}
    if eval_after:
        print("\n--- Evaluating AFTER pruning ---")
        after_metrics = evaluate(
            model,
            model_config_cls,
            val_loader,
            device,
            max_samples=args.max_eval_samples,
        )
        psnr = after_metrics.get("val_psnr", float("nan"))
        ssim = after_metrics.get("val_ssim", float("nan"))
        lpips = after_metrics.get("val_lpips", float("nan"))
        print(f"  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}  LPIPS={lpips:.4f}")

    # Print comparison
    if before_metrics and after_metrics:
        if is_fair:
            before_metrics["active_codes"] = original_codebook
            after_metrics["active_codes"] = new_codebook
        else:
            before_metrics["active_codes"] = original_K
            after_metrics["active_codes"] = quantizer.K
        print_comparison(before_metrics, after_metrics)

    # Save pruned checkpoint
    output_path = args.output
    if output_path is None:
        p = Path(args.checkpoint)
        output_path = str(p.parent / f"{p.stem}_pruned{p.suffix}")

    # Update config to reflect pruned size
    pruned_config = dict(config)
    if is_fair:
        pruned_config["lmb_levels"] = quantizer.levels
        pruned_config["lgq_levels"] = quantizer.levels
    else:
        pruned_config["num_bins"] = quantizer.K

    pruned_checkpoint = {
        "epoch": checkpoint.get("epoch"),
        "model_state_dict": model.state_dict(),
        "config": pruned_config,
        "model_type": checkpoint.get("model_type"),
        "run_name": checkpoint.get("run_name", "") + "_pruned",
        "pruning_info": {
            "strategy": args.strategy,
            "original_K": original_K if not is_fair else None,
            "pruned_K": quantizer.K if not is_fair else None,
            "original_levels": original_levels if is_fair else None,
            "pruned_levels": quantizer.levels if is_fair else None,
            "source_checkpoint": str(args.checkpoint),
            "before_metrics": before_metrics,
            "after_metrics": after_metrics,
        },
    }

    torch.save(pruned_checkpoint, output_path)
    print(f"\nSaved pruned checkpoint: {output_path}")

    # File size comparison
    orig_size = Path(args.checkpoint).stat().st_size / (1024 * 1024)
    pruned_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {orig_size:.1f}MB -> {pruned_size:.1f}MB ({(1 - pruned_size/orig_size)*100:.1f}% smaller)")


if __name__ == "__main__":
    main()

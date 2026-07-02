#!/usr/bin/env python3
"""
Visualize reconstruction comparison across FSQ, LMB, SIM_VQ, and VQ.

Loads one batch of validation images, runs each model to get reconstructions,
and saves a grid image: rows = samples, columns = Original | FSQ | LMB | SIM_VQ | VQ.

Usage:
    python scripts/visualize_reconstruction_comparison.py --data-root data/imagenet
    python scripts/visualize_reconstruction_comparison.py --checkpoint-fsq path.pt --checkpoint-lmb path.pt ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Default checkpoint paths (16K codebook main runs)
DEFAULT_CHECKPOINTS = {
    "fsq": "results/fsq/fsq_lv16-16-8-8_bs32_lr3e-4_dim128_20260123_184222_deab/checkpoints/latest_model.pt",
    "lmb": "results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd/checkpoints/latest_model.pt",
    "sim_vq": "results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32/checkpoints/latest_model.pt",
    "vq": "results/vq/vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0/checkpoints/latest_model.pt",
}

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def get_transform(image_size: int = 128):
    """Transform that resizes to image_size and normalizes (ImageNet stats)."""
    from PIL import Image
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _transform(pil_img):
        img = pil_img.resize((image_size, image_size), Image.LANCZOS)
        x = np.array(img).astype(np.float32) / 255.0
        x = (x - mean) / std
        return torch.from_numpy(x).permute(2, 0, 1)

    return _transform


def load_dataset(data_root: str, split: str = "val", image_size: int = 128, max_images: int = 32):
    """Load ImageNet-style dataset (ImageFolder or flat directory)."""
    from torchvision import datasets
    data_path = Path(data_root) / split
    if not data_path.exists() and split == "val":
        data_path = Path(data_root) / "test"
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path} (tried val/ and test/)")

    transform = get_transform(image_size)
    try:
        dataset = datasets.ImageFolder(root=str(data_path), transform=transform)
    except Exception:
        from scripts.visualize_latent_vs_codes import FlatImageDataset
        dataset = FlatImageDataset(str(data_path), transform=transform, max_images=max_images)

    return dataset


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load model (same logic as evaluate.py)."""
    from scripts.evaluation.evaluate import load_model_from_checkpoint as load_ckpt
    model, _model_config, _args, _ckpt = load_ckpt(checkpoint_path, device)
    return model


@torch.no_grad()
def get_reconstructions(model, x: torch.Tensor, device: str) -> torch.Tensor:
    """Run model forward and return reconstructions (same shape as x)."""
    x = x.to(device)
    out = model(x)
    recon = out[0]
    return recon.cpu()


def build_grid(originals: torch.Tensor, recons: dict, num_rows: int = 4) -> torch.Tensor:
    """
    Build a grid image: each row is one sample, columns are Original | FSQ | LMB | SIM_VQ | VQ.
    originals: [N, 3, H, W], recons: {"fsq": [N,3,H,W], "lmb": ..., "sim_vq": ..., "vq": ...}
    Returns [1, 3, H_grid, W_grid] or single image tensor for save_image.
    """
    order = ["Original", "FSQ", "LMB", "SIM_VQ", "VQ"]
    n_cols = len(order)
    n_rows = min(num_rows, originals.size(0))
    # Per-cell size
    h, w = originals.shape[2], originals.shape[3]
    pad = 2
    cell_h, cell_w = h + pad * 2, w + pad * 2
    grid_h = n_rows * cell_h
    grid_w = n_cols * cell_w
    grid = torch.ones(3, grid_h, grid_w)
    # Fill
    for row in range(n_rows):
        # Original
        img = originals[row].clamp(0, 1)
        grid[:, row * cell_h + pad : row * cell_h + pad + h, 0 * cell_w + pad : 0 * cell_w + pad + w] = img
        for col, key in enumerate(["fsq", "lmb", "sim_vq", "vq"], start=1):
            if key in recons and recons[key] is not None:
                img = recons[key][row].clamp(0, 1)
                grid[:, row * cell_h + pad : row * cell_h + pad + h, col * cell_w + pad : col * cell_w + pad + w] = img
    return grid


def main():
    parser = argparse.ArgumentParser(description="Reconstruction comparison: Original | FSQ | LMB | SIM_VQ | VQ")
    parser.add_argument("--data-root", type=str, default="data/imagenet", help="ImageNet root (contains val/)")
    parser.add_argument("--output", type=str, default="results/plots/reconstruction_comparison_fsq_lmb_simvq_vq.png",
                        help="Output image path")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of sample images (rows in grid)")
    parser.add_argument("--image-size", type=int, default=128, help="Model input size")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checkpoint-fsq", type=str, default=None)
    parser.add_argument("--checkpoint-lmb", type=str, default=None)
    parser.add_argument("--checkpoint-sim-vq", type=str, default=None)
    parser.add_argument("--checkpoint-vq", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    root = Path(PROJECT_ROOT)
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint paths
    checkpoints = {
        "fsq": args.checkpoint_fsq or str(root / DEFAULT_CHECKPOINTS["fsq"]),
        "lmb": args.checkpoint_lmb or str(root / DEFAULT_CHECKPOINTS["lmb"]),
        "sim_vq": args.checkpoint_sim_vq or str(root / DEFAULT_CHECKPOINTS["sim_vq"]),
        "vq": args.checkpoint_vq or str(root / DEFAULT_CHECKPOINTS["vq"]),
    }

    # Load dataset
    print("Loading dataset...")
    try:
        dataset = load_dataset(args.data_root, split="val", image_size=args.image_size, max_images=args.num_samples)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Use --data-root to point to ImageNet root (e.g. data/imagenet).")
        return 1
    loader = DataLoader(dataset, batch_size=args.num_samples, shuffle=True, num_workers=0)
    batch_x, _ = next(iter(loader))

    # Denormalize for display (store originals in [0,1])
    originals = (batch_x * STD + MEAN).clamp(0, 1)

    # Load models and get reconstructions
    recons = {"fsq": None, "lmb": None, "sim_vq": None, "vq": None}
    for name, ckpt_path in checkpoints.items():
        path = Path(ckpt_path)
        if not path.exists():
            print(f"  Skip {name}: checkpoint not found: {path}")
            continue
        print(f"  Loading {name}: {path.name}")
        try:
            model = load_model_from_checkpoint(str(path), device)
            recons[name] = get_reconstructions(model, batch_x, device)
            recons[name] = (recons[name] * STD + MEAN).clamp(0, 1)
        except Exception as e:
            print(f"  Skip {name}: {e}")
            recons[name] = None

    # Build grid
    num_rows = min(4, args.num_samples)
    grid = build_grid(originals, recons, num_rows=num_rows)

    # Save grid with column labels (Original | FSQ | LMB | SIM_VQ | VQ)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arr = grid.permute(1, 2, 0).numpy().clip(0, 1)
    fig, ax = plt.subplots(1, 1, figsize=(14, 3.5 * num_rows))
    ax.imshow(arr)
    ax.set_axis_off()
    cols = ["Original", "FSQ", "LMB", "SIM_VQ", "VQ"]
    cell_w = arr.shape[1] / 5
    for i, label in enumerate(cols):
        ax.text(cell_w * (i + 0.5), -12, label, ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

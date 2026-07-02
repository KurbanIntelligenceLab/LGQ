#!/usr/bin/env python3
"""
LFQ-specific UMAP visualization: encoder outputs (z) vs active codebook entries (e).

LFQ uses a different embedding architecture than FSQ:
- Spherical mode (default): L2-normalizes input before quantization onto the unit hypersphere
- Codes are also L2-normalized
- Both z and e must be in this spherical space for correct comparison

Usage:
    python scripts/visualize_latent_vs_codes_lfq.py
    python scripts/visualize_latent_vs_codes_lfq.py --checkpoint results/lfq/.../latest_model.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader

DEFAULT_LFQ_CHECKPOINT = "results/lfq/lfq_cb16384_bs32_lr3e-4_dim128_20260123_221044_6f79/checkpoints/latest_model.pt"


class FlatImageDataset(Dataset):
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff"}

    def __init__(self, root: str, transform=None, max_images: int | None = None):
        root = Path(root)
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(root.glob("*" + ext))
            paths.extend(root.glob("*" + ext.upper()))
            paths.extend(root.glob("**/*" + ext))
            paths.extend(root.glob("**/*" + ext.upper()))
        self.paths = sorted(set(paths))
        if max_images:
            self.paths = self.paths[:max_images]
        if not self.paths:
            raise RuntimeError(f"No images in {root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = Image.open(self.paths[i]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_transform(image_size: int = 128):
    def _transform(img):
        img = img.resize((image_size, image_size), Image.LANCZOS)
        x = np.array(img).astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (x - mean) / std
    return _transform


def load_model(checkpoint_path: str, device: str):
    from configs import MODEL_CONFIGS
    from quantization.model import ModelConfig

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt["config"]
    model_type = ckpt.get("model_type", config.get("model"))
    if model_type != "lfq":
        raise ValueError(f"Expected LFQ checkpoint, got {model_type}")

    model_cfg = ModelConfig(
        base_ch=config.get("dim", 128) // 2,
        embedding_dim=config.get("embedding_dim", 128),
    )
    model = MODEL_CONFIGS["lfq"].create_model(argparse.Namespace(**config), model_cfg)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, config


@torch.no_grad()
def collect_z_and_zq_embedding(
    model,
    dataloader: DataLoader,
    device: str,
    num_images: int = 1000,
    subsample_latents: int = 15000,
):
    """
    Collect z (encoder output) and z_q (actual quantized output) in embedding_dim.
    Both from the same forward pass - compares what encoder produces vs what
    quantizer produces, in the space the decoder sees.
    e = unique z_q values (the 'active codes' = actual quantized representations used).
    """
    model.eval()
    all_z = []
    all_zq = []
    images_done = 0

    for x, _ in tqdm(dataloader, desc="Collecting LFQ", total=min(len(dataloader), num_images // dataloader.batch_size + 1)):
        if images_done >= num_images:
            break
        x = x.to(device)
        images_done += x.size(0)

        z = model.enc(x)
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        z_quant = model.pre_quant(z_norm)
        z_input = F.normalize(z_quant.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1), dim=-1)

        # Run full quant path to get z_q
        z_norm_bchw = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        z_norm_bchw = model.pre_quant(z_norm_bchw)
        z_q, _, _ = model.quantize(z_norm_bchw)
        z_q = model.post_quant(z_q)  # [B, embedding_dim, H, W]
        z_q_flat = z_q.permute(0, 2, 3, 1).contiguous().view(B * H * W, -1)

        all_z.append(z_norm.cpu())
        all_zq.append(z_q_flat.cpu())

    z_full = torch.cat(all_z, dim=0)
    zq_full = torch.cat(all_zq, dim=0)
    n_total = z_full.size(0)

    if n_total > subsample_latents:
        rng = torch.Generator().manual_seed(42)
        perm = torch.randperm(n_total, generator=rng)[:subsample_latents]
        z = z_full[perm]
        zq_sub = zq_full[perm]
    else:
        z = z_full
        zq_sub = zq_full

    # Unique z_q values (active codes = actual quantized vectors used)
    zq_np = zq_sub.numpy()
    unique_rows, inverse = np.unique(zq_np.round(decimals=6), axis=0, return_inverse=True)
    e = torch.from_numpy(unique_rows.astype(np.float32))
    n_active = e.size(0)

    print(f"  Tokens: {n_total}, Subsampled: {z.size(0)}, Unique z_q (active codes): {n_active}")
    return z, e, n_active


def run_umap(z: torch.Tensor, e: torch.Tensor, n_neighbors: int = 25, min_dist: float = 0.15):
    import umap
    combined = torch.cat([z, e], dim=0).detach().numpy()
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric="euclidean", random_state=42, verbose=False)
    emb = reducer.fit_transform(combined)
    nz = z.size(0)
    return emb[:nz], emb[nz:]


def compute_metrics(z: torch.Tensor, e: torch.Tensor):
    from sklearn.neighbors import NearestNeighbors
    z_np = z.detach().numpy() if isinstance(z, torch.Tensor) else z
    e_np = e.detach().numpy() if isinstance(e, torch.Tensor) else e
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(e_np)
    d, _ = nn.kneighbors(z_np)
    latent_to_code = d.flatten()
    nn2 = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(z_np)
    d2, _ = nn2.kneighbors(e_np)
    code_to_latent = d2.flatten()
    z_std = float(np.std(z_np) + 1e-8)
    orphan = 100.0 * (code_to_latent > 2 * latent_to_code.mean()).mean()
    return {"relative_quant_error": float(latent_to_code.mean() / z_std), "orphan_codes_pct": float(orphan)}


def main():
    ap = argparse.ArgumentParser(description="LFQ-specific z vs e UMAP (spherical space)")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--data-root", type=str, default="data/imagenet")
    ap.add_argument("--num-images", type=int, default=1000)
    ap.add_argument("--subsample-latents", type=int, default=15000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--output-dir", type=str, default="plots")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint or DEFAULT_LFQ_CHECKPOINT)
    if not ckpt_path.is_absolute():
        ckpt_path = project_root / ckpt_path
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"LFQ checkpoint: {ckpt_path}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root).expanduser()
    test_path = data_root / "test" if (data_root / "test").exists() else data_root / "val"
    transform = get_transform(128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model, config = load_model(str(ckpt_path), device)
    z, e, n_active = collect_z_and_zq_embedding(
        model, loader, device,
        num_images=args.num_images,
        subsample_latents=args.subsample_latents,
    )

    print("\nRunning UMAP (z vs z_q in embedding space)...")
    z_emb, e_emb = run_umap(z, e)
    metrics = compute_metrics(z, e)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(z_emb[:, 0], z_emb[:, 1], c="#0066cc", alpha=0.35, s=4, label="Encoder output (z)", rasterized=True, zorder=1)
    max_red = 3200
    e_plot = e_emb if e_emb.shape[0] <= max_red else e_emb[np.random.default_rng(42).choice(e_emb.shape[0], max_red, replace=False)]
    ax.scatter(e_plot[:, 0], e_plot[:, 1], c="#e74c3c", s=8, alpha=0.4, marker="o", label=f"Quantized (z_q), n={e_plot.shape[0]:,} of {n_active:,}", rasterized=True, edgecolors="#8B0000", linewidths=0.3, zorder=2)
    err, orphan = metrics["relative_quant_error"], metrics["orphan_codes_pct"]
    ax.set_title(f"LFQ 16K (embedding space)\nUnique z_q: {n_active:,}  |  Rel. quant error: {err:.3f}  |  Orphan %: {orphan:.1f}", fontsize=10)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_aspect("equal", adjustable="datalim")
    plt.suptitle("LFQ: Encoder output (z) vs quantized output (z_q) — same forward pass, embedding space\nBlue = pre-quantization, Red = post-quantization (what decoder receives)", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = out_dir / "latent_vs_codes_lfq_spherical.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out_path}")

    with open(out_dir / "latent_vs_codes_lfq_metrics.txt", "w") as f:
        f.write(f"LFQ 16K (embedding space: z vs z_q)\nUnique z_q: {n_active}\n")
        f.write(f"Relative quant error: {metrics['relative_quant_error']:.4f}\nOrphan %: {metrics['orphan_codes_pct']:.2f}\n")
    print("Done.")


if __name__ == "__main__":
    main()

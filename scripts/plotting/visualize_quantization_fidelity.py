#!/usr/bin/env python3
"""
Visualize quantization fidelity: how well do quantized outputs match encoder outputs?

This compares the QUANTIZED representation (after full quantization pipeline)
to the ORIGINAL encoder outputs, showing how well each method preserves
the latent distribution.

Usage:
    python scripts/visualize_quantization_fidelity.py \
        --checkpoint1 results/sim_vq/.../best_model.pt \
        --checkpoint2 results/fsq/.../best_model.pt \
        --num-images 1000
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "external"))

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class FlatImageDataset(Dataset):
    """Simple image dataset that loads all images from a directory."""
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff'}
    
    def __init__(self, root: str, transform=None, max_images=None):
        self.root = root
        self.transform = transform
        self.image_paths = []
        root_path = Path(root)
        for ext in self.EXTENSIONS:
            self.image_paths.extend(root_path.glob(f'*{ext}'))
            self.image_paths.extend(root_path.glob(f'*{ext.upper()}'))
        self.image_paths = sorted(set(self.image_paths))
        if max_images:
            self.image_paths = self.image_paths[:max_images]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


def get_transform(image_size=128):
    """Get image transform matching training."""
    def transform(img):
        img = img.resize((image_size, image_size), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = (img - mean) / std
        return img
    return transform


def load_model_from_checkpoint(checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    from configs import MODEL_CONFIGS
    from quantization.model import ModelConfig
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model_type = checkpoint.get("model_type", config.get("model"))
    
    model_config_cls = MODEL_CONFIGS[model_type]
    args = argparse.Namespace(**config)
    
    model_cfg = ModelConfig(
        base_ch=config.get("dim", 128) // 2,
        embedding_dim=config.get("embedding_dim", 128),
    )
    
    model = model_config_cls.create_model(args, model_cfg)
    
    state_dict = checkpoint["model_state_dict"].copy()
    if "quantize.centers" in state_dict:
        centers = state_dict["quantize.centers"]
        if hasattr(model, 'quantize') and hasattr(model.quantize, 'centers'):
            if model.quantize.centers.shape != centers.shape:
                if centers.shape[0] == model.quantize.centers.shape[1]:
                    state_dict["quantize.centers"] = centers.t()
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    epoch = checkpoint.get('epoch', 'N/A')
    return model, config, model_type, epoch


@torch.no_grad()
def collect_encoder_and_quantized(model, dataloader, device, num_images=1000, subsample=15000):
    """
    Collect original encoder outputs and quantized outputs.
    
    Returns encoder outputs (z) and quantized outputs (z_q), both in 128D.
    """
    model.eval()
    
    all_z = []  # Original encoder outputs (before quantization)
    all_z_q = []  # Quantized outputs (after full pipeline, projected back to 128D)
    all_indices = set()
    images_processed = 0
    
    pbar = tqdm(dataloader, desc="Collecting", total=min(len(dataloader), num_images // dataloader.batch_size + 1))
    for x, _ in pbar:
        if images_processed >= num_images:
            break
        
        x = x.to(device)
        images_processed += x.size(0)
        
        # Get encoder output
        z = model.enc(x)  # [B, C, H, W]
        B, C, H, W = z.shape
        
        # Get quantized output from full forward pass
        outputs = model(x)
        recon = outputs[0]  # Reconstruction
        indices = outputs[1]
        
        # We need the quantized latent BEFORE decoder
        # Re-run the quantization to get z_q
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat)
        
        # Get quantized representation based on model type
        if model.quantizer_type == "fsq":
            z_pre = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_pre = model.pre_quant(z_pre)  # [B, 4, H, W]
            z_q_small, _ = model.quantize(z_pre)  # [B, 4, H, W]
            z_q = model.post_quant(z_q_small)  # [B, 128, H, W]
            z_q_flat = z_q.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        elif model.quantizer_type == "sim_vq":
            z_pre = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_q, _, _ = model.quantize(z_pre)  # [B, 128, H, W]
            z_q_flat = z_q.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        elif model.quantizer_type == "vq":
            z_pre = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_q, _, _ = model.quantize(z_pre)  # [B, 128, H, W]
            z_q_flat = z_q.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        elif model.quantizer_type == "lmb":
            if model.pre_quant is not None:
                z_pre = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                z_pre = model.pre_quant(z_pre)  # Project to smaller dim
                z_bct = z_pre.view(B, -1, H * W)
                _, z_q_bct, *_ = model.quantize(z_bct)
                z_q_small = z_q_bct.view(B, -1, H, W)
                z_q = model.post_quant(z_q_small)
                z_q_flat = z_q.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            else:
                z_bct = z_norm.view(B, H * W, C).permute(0, 2, 1).contiguous()
                _, z_q_bct, *_ = model.quantize(z_bct)
                z_q_flat = z_q_bct.permute(0, 2, 1).contiguous().view(B * H * W, C)
        else:
            # Fallback
            z_q_flat = z_norm
        
        all_z.append(z_norm.cpu())
        all_z_q.append(z_q_flat.cpu())
        all_indices.update(indices.view(-1).cpu().numpy().tolist())
        
        pbar.set_postfix({"images": images_processed, "active": len(all_indices)})
    
    all_z = torch.cat(all_z, dim=0)
    all_z_q = torch.cat(all_z_q, dim=0)
    
    # Subsample
    if all_z.shape[0] > subsample:
        idx = torch.randperm(all_z.shape[0])[:subsample]
        all_z = all_z[idx]
        all_z_q = all_z_q[idx]
    
    return all_z, all_z_q, len(all_indices)


def run_umap_combined(z1, zq1, z2, zq2, n_neighbors=15, min_dist=0.1):
    """Run UMAP on all data combined for consistent embedding."""
    import umap
    
    # Stack all data
    all_data = torch.cat([z1, zq1, z2, zq2], dim=0).numpy()
    n1, n2 = z1.shape[0], z2.shape[0]
    
    print(f"  Running UMAP on {all_data.shape[0]} points...")
    
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        metric='euclidean',
        random_state=42,
        verbose=False
    )
    embedding = reducer.fit_transform(all_data)
    
    # Split back
    z1_emb = embedding[:n1]
    zq1_emb = embedding[n1:2*n1]
    z2_emb = embedding[2*n1:2*n1+n2]
    zq2_emb = embedding[2*n1+n2:]
    
    return z1_emb, zq1_emb, z2_emb, zq2_emb


def compute_fidelity_metrics(z, z_q):
    """Compute how well quantized outputs match original encoder outputs."""
    z_np = z.numpy() if isinstance(z, torch.Tensor) else z
    zq_np = z_q.numpy() if isinstance(z_q, torch.Tensor) else z_q
    
    # Per-sample quantization error
    errors = np.linalg.norm(z_np - zq_np, axis=1)
    
    # Normalize by data spread
    z_std = z_np.std()
    
    return {
        'quant_error_mean': float(errors.mean()),
        'quant_error_median': float(np.median(errors)),
        'quant_error_std': float(errors.std()),
        'relative_error': float(errors.mean() / z_std),
        'z_std': float(z_std),
    }


def plot_comparison(z1_emb, zq1_emb, z2_emb, zq2_emb, 
                   metrics1, metrics2, 
                   name1, name2, epoch1, epoch2,
                   active1, active2,
                   output_path):
    """Create comparison plot showing original vs quantized distributions."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Model 1 plot
    ax = axes[0]
    ax.scatter(z1_emb[:, 0], z1_emb[:, 1], c='#3498db', alpha=0.85, s=6,
               label='Original encoder output', rasterized=True)
    ax.scatter(zq1_emb[:, 0], zq1_emb[:, 1], c='#e74c3c', alpha=0.05, s=2, marker='o',
               label='Quantized output', rasterized=True)
    ax.set_title(f'{name1} (epoch {epoch1})\n'
                 f'Active codes: {active1:,}\n'
                 f'Rel. quant error: {metrics1["relative_error"]:.3f}',
                 fontsize=11)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper right', fontsize=9)
    
    # Model 2 plot  
    ax = axes[1]
    ax.scatter(z2_emb[:, 0], z2_emb[:, 1], c='#3498db', alpha=0.85, s=6,
               label='Original encoder output', rasterized=True)
    ax.scatter(zq2_emb[:, 0], zq2_emb[:, 1], c='#e74c3c', alpha=0.05, s=2, marker='o',
               label='Quantized output', rasterized=True)
    ax.set_title(f'{name2} (epoch {epoch2})\n'
                 f'Active codes: {active2:,}\n'
                 f'Rel. quant error: {metrics2["relative_error"]:.3f}',
                 fontsize=11)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.suptitle('Quantization Fidelity: Original vs Quantized Encoder Outputs\n'
                 'Blue = original encoder output, Red = after quantization (lower overlap = more distortion)',
                 fontsize=13, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\nSaved plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize quantization fidelity")
    parser.add_argument("--checkpoint1", type=str, required=True, help="First model checkpoint")
    parser.add_argument("--checkpoint2", type=str, required=True, help="Second model checkpoint")
    parser.add_argument("--data-root", type=str, default="data/imagenet")
    parser.add_argument("--num-images", type=int, default=1000)
    parser.add_argument("--subsample", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_path = Path(args.data_root).expanduser()
    test_path = data_path / "test"
    if not test_path.exists():
        test_path = data_path / "val"
    
    print(f"\nLoading images from: {test_path}")
    transform = get_transform(image_size=128)
    dataset = FlatImageDataset(str(test_path), transform=transform, max_images=args.num_images * 2)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Load and process model 1
    print(f"\n{'='*60}")
    print(f"Loading model 1: {args.checkpoint1.split('/')[-3]}")
    print(f"{'='*60}")
    model1, config1, type1, epoch1 = load_model_from_checkpoint(args.checkpoint1, device)
    print(f"  Type: {type1}, Epoch: {epoch1}")
    
    z1, zq1, active1 = collect_encoder_and_quantized(model1, dataloader, device, args.num_images, args.subsample)
    print(f"  Collected {z1.shape[0]} samples, {active1} active codes")
    
    del model1
    
    # Load and process model 2
    print(f"\n{'='*60}")
    print(f"Loading model 2: {args.checkpoint2.split('/')[-3]}")
    print(f"{'='*60}")
    model2, config2, type2, epoch2 = load_model_from_checkpoint(args.checkpoint2, device)
    print(f"  Type: {type2}, Epoch: {epoch2}")
    
    z2, zq2, active2 = collect_encoder_and_quantized(model2, dataloader, device, args.num_images, args.subsample)
    print(f"  Collected {z2.shape[0]} samples, {active2} active codes")
    
    del model2
    
    # Compute metrics
    print(f"\n{'='*60}")
    print("Computing fidelity metrics...")
    print(f"{'='*60}")
    
    metrics1 = compute_fidelity_metrics(z1, zq1)
    metrics2 = compute_fidelity_metrics(z2, zq2)
    
    print(f"\n{type1.upper()} metrics:")
    for k, v in metrics1.items():
        print(f"  {k}: {v:.4f}")
    
    print(f"\n{type2.upper()} metrics:")
    for k, v in metrics2.items():
        print(f"  {k}: {v:.4f}")
    
    # Run UMAP
    print(f"\n{'='*60}")
    print("Running UMAP...")
    print(f"{'='*60}")
    
    z1_emb, zq1_emb, z2_emb, zq2_emb = run_umap_combined(z1, zq1, z2, zq2)
    
    # Create plot
    print(f"\n{'='*60}")
    print("Creating visualization...")
    print(f"{'='*60}")
    
    output_path = output_dir / f"quantization_fidelity_{type1}_vs_{type2}.png"
    plot_comparison(
        z1_emb, zq1_emb, z2_emb, zq2_emb,
        metrics1, metrics2,
        type1.upper(), type2.upper(),
        epoch1, epoch2,
        active1, active2,
        output_path
    )
    
    # Save metrics
    metrics_path = output_dir / f"quantization_fidelity_{type1}_vs_{type2}.txt"
    with open(metrics_path, 'w') as f:
        f.write("Quantization Fidelity Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Images: {args.num_images}, Samples: {z1.shape[0]}\n\n")
        
        f.write(f"{type1.upper()} (epoch {epoch1}):\n")
        f.write(f"  Active codes: {active1}\n")
        for k, v in metrics1.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write(f"\n{type2.upper()} (epoch {epoch2}):\n")
        f.write(f"  Active codes: {active2}\n")
        for k, v in metrics2.items():
            f.write(f"  {k}: {v:.4f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("INTERPRETATION:\n")
        f.write("Lower relative_error = quantized output closer to original encoder output\n")
        f.write("In the plot, more overlap between blue/red = better preservation of latent distribution\n")
    
    print(f"Saved metrics to: {metrics_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()

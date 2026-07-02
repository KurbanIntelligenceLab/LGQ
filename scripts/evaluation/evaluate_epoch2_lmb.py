#!/usr/bin/env python3
"""
Evaluate LMB flattened model for epoch 2 and create plots.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Import model components
from configs import MODEL_CONFIGS
from configs.base import ModelConfig
from scripts.evaluation.metrics import MetricsCalculator, compute_codebook_utilization
from scripts.evaluation.fid import FIDCalculator

def load_model_from_checkpoint(checkpoint_path, config_path, device='cuda'):
    """Load model from checkpoint."""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Create a Namespace-like object from config for model creation
    from argparse import Namespace
    args = Namespace()
    args.model = config.get('model', 'lmb')
    args.dim = config.get('dim', 128)
    args.embedding_dim = config.get('embedding_dim', 7)
    args.num_bins = config.get('num_bins', 16384)
    args.flatten_channels = config.get('flatten_channels', True)
    args.tau_start = config.get('tau_start', 1.0)
    args.tau_end = config.get('tau_end', 0.1)
    args.learn_widths = config.get('learn_widths', False)
    
    # Get model config class
    model_config_class = MODEL_CONFIGS[args.model]
    
    # Create model config
    model_config = ModelConfig(
        base_ch=args.dim,
        embedding_dim=args.embedding_dim
    )
    
    # Create model using the config class
    model = model_config_class.create_model(args, model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config

def evaluate_model(model, dataloader, device='cuda', num_samples=None):
    """Evaluate model on validation set."""
    model.eval()
    
    metrics_calc = MetricsCalculator(device=device)
    fid_calc = FIDCalculator(device=device)
    
    all_reconstructions = []
    all_originals = []
    all_indices = []
    
    total_rec_loss = 0
    total_vq_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if num_samples and batch_idx * dataloader.batch_size >= num_samples:
                break
                
            images = images.to(device)
            
            # Forward pass
            if hasattr(model, 'quantizer') and hasattr(model.quantizer, 'flatten_channels') and model.quantizer.flatten_channels:
                # LMB with flatten_channels
                recon, indices, vq_loss, perplexity = model(images)
            else:
                # Standard LMB
                recon, indices, vq_loss, perplexity = model(images)
            
            # Compute losses
            rec_loss = (recon - images).abs().mean()
            total_rec_loss += rec_loss.item()
            total_vq_loss += vq_loss.item()
            num_batches += 1
            
            # Collect for metrics
            all_reconstructions.append(recon.cpu())
            all_originals.append(images.cpu())
            all_indices.append(indices.cpu())
    
    # Concatenate all batches
    all_reconstructions = torch.cat(all_reconstructions, dim=0)
    all_originals = torch.cat(all_originals, dim=0)
    all_indices = torch.cat(all_indices, dim=0)
    
    # Compute metrics
    print("Computing image quality metrics...")
    psnr = metrics_calc.compute_batch_metrics(
        all_originals.to(device),
        all_reconstructions.to(device),
        compute_lpips=True
    )
    
    print("Computing FID...")
    rfid = fid_calc.compute_rfid(
        all_originals.to(device),
        all_reconstructions.to(device)
    )
    
    # Compute codebook metrics
    codebook_size = 16384  # From config
    codebook_metrics = compute_codebook_utilization(
        all_indices.to(device),
        codebook_size
    )
    
    avg_rec_loss = total_rec_loss / num_batches
    avg_vq_loss = total_vq_loss / num_batches
    
    return {
        'rec_loss': avg_rec_loss,
        'vq_loss': avg_vq_loss,
        'psnr': psnr['psnr'],
        'ssim': psnr['ssim'],
        'lpips': psnr['lpips'],
        'rfid': rfid,
        'active_codes': codebook_metrics['active_codes'],
        'codebook_util': codebook_metrics['codebook_utilization'],
        'perplexity': codebook_metrics['perplexity'],
    }

def main():
    # Paths
    run_dir = Path("results/lmb/lmb_nb16384_tau1.0-0.1_bs64_lr3e-4_dim128_20260119_214749_6c81")
    checkpoint_path = run_dir / "checkpoints" / "best_model.pt"
    config_path = run_dir / "config.json"
    eval_metrics_path = run_dir / "eval_metrics.csv"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, config = load_model_from_checkpoint(checkpoint_path, config_path, device)
    
    # Load validation dataset
    print("Loading validation dataset...")
    data_root = config.get('data_root', 'data/imagenet')
    
    class ImageFolderDataset(Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.samples = []
            for class_dir in os.listdir(root):
                class_path = os.path.join(root, class_dir)
                if os.path.isdir(class_path):
                    for img_name in os.listdir(class_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append((os.path.join(class_path, img_name), 0))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
    
    def transform_fn(img):
        import torch
        from torchvision.transforms.functional import resize, center_crop, to_tensor
        img = resize(img, 256)
        img = center_crop(img, 256)
        return to_tensor(img)
    
    val_dataset = ImageFolderDataset(
        root=os.path.join(data_root, 'val'),
        transform=transform_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )
    
    # Evaluate
    print("Running evaluation...")
    metrics = evaluate_model(model, val_loader, device, num_samples=5000)
    
    # Determine epoch (check if epoch 2 checkpoint exists, otherwise use epoch 2)
    epoch = 2
    global_step = 32550  # From train_metrics.csv
    
    # Load existing eval metrics
    existing_metrics = []
    if eval_metrics_path.exists():
        with open(eval_metrics_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_metrics = list(reader)
    
    # Add new metrics
    new_row = {
        'epoch': str(epoch),
        'global_step': str(global_step),
        'val_active_codes': str(int(metrics['active_codes'])),
        'val_codebook_util': str(metrics['codebook_util']),
        'val_lpips': str(metrics['lpips']),
        'val_perplexity': str(metrics['perplexity']),
        'val_psnr': str(metrics['psnr']),
        'val_rec_loss': str(metrics['rec_loss']),
        'val_rfid': str(metrics['rfid']),
        'val_ssim': str(metrics['ssim']),
        'val_total_loss': str(metrics['rec_loss'] + metrics['vq_loss']),
        'val_vq_loss': str(metrics['vq_loss']),
    }
    
    # Write to CSV
    fieldnames = ['epoch', 'global_step', 'val_active_codes', 'val_codebook_util', 
                  'val_lpips', 'val_perplexity', 'val_psnr', 'val_rec_loss', 
                  'val_rfid', 'val_ssim', 'val_total_loss', 'val_vq_loss']
    
    file_exists = eval_metrics_path.exists()
    with open(eval_metrics_path, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(new_row)
    
    print(f"\nEvaluation complete! Results saved to {eval_metrics_path}")
    print(f"\nEpoch {epoch} Metrics:")
    print(f"  rFID: {metrics['rfid']:.2f}")
    print(f"  PSNR: {metrics['psnr']:.2f}")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  LPIPS: {metrics['lpips']:.4f}")
    print(f"  Rec Loss: {metrics['rec_loss']:.4f}")
    print(f"  Active Codes: {int(metrics['active_codes'])} ({metrics['codebook_util']:.2f}%)")
    print(f"  Perplexity: {metrics['perplexity']:.2f}")
    
    # Create plots
    print("\nCreating plots...")
    create_plots(run_dir)
    print("Done!")

def create_plots(run_dir):
    """Create visualization plots for the metrics."""
    eval_path = run_dir / "eval_metrics.csv"
    train_path = run_dir / "train_metrics.csv"
    
    if not eval_path.exists():
        print("No eval metrics found for plotting")
        return
    
    # Load eval metrics
    epochs = []
    rfid = []
    psnr = []
    ssim = []
    lpips = []
    rec_loss = []
    active_codes = []
    
    with open(eval_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            rfid.append(float(row['val_rfid']))
            psnr.append(float(row['val_psnr']))
            ssim.append(float(row['val_ssim']))
            lpips.append(float(row['val_lpips']))
            rec_loss.append(float(row['val_rec_loss']))
            active_codes.append(int(row['val_active_codes']))
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('LMB Flattened Model - Evaluation Metrics', fontsize=16, fontweight='bold')
    
    # Plot 1: rFID
    axes[0, 0].plot(epochs, rfid, 'o-', linewidth=2, markersize=8, color='#1f77b4')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('rFID (lower is better)')
    axes[0, 0].set_title('Reconstruction FID')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: PSNR
    axes[0, 1].plot(epochs, psnr, 'o-', linewidth=2, markersize=8, color='#ff7f0e')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (higher is better)')
    axes[0, 1].set_title('Peak Signal-to-Noise Ratio')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: SSIM
    axes[0, 2].plot(epochs, ssim, 'o-', linewidth=2, markersize=8, color='#2ca02c')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('SSIM (higher is better)')
    axes[0, 2].set_title('Structural Similarity Index')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: LPIPS
    axes[1, 0].plot(epochs, lpips, 'o-', linewidth=2, markersize=8, color='#d62728')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('LPIPS (lower is better)')
    axes[1, 0].set_title('Learned Perceptual Image Patch Similarity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Rec Loss
    axes[1, 1].plot(epochs, rec_loss, 'o-', linewidth=2, markersize=8, color='#9467bd')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Reconstruction Loss')
    axes[1, 1].set_title('Reconstruction Loss')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Active Codes
    axes[1, 2].plot(epochs, active_codes, 'o-', linewidth=2, markersize=8, color='#8c564b')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Active Codes')
    axes[1, 2].set_title('Codebook Utilization')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path("plots") / "lmb_flattened_metrics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    plt.close()

if __name__ == "__main__":
    main()

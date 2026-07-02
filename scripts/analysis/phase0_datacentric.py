#!/usr/bin/env python3
"""
Phase 0: Data-Centric Codebook Initialisation & N* Prediction.

Runs BEFORE any LGQ training to produce an optimal codebook initialisation
based on Gersho-Zador quantisation theory.

Steps:
  1. Encode M' image patches through frozen encoder -> latent vectors Z
  2. Estimate latent covariance, compute effective intrinsic dimension d_eff
  3. Predict optimal codebook size N* via Gersho-Zador formula
  4. Density-weighted k-means initialisation (Gersho-optimal density)
  5. Utilisation analysis of all initialisation strategies
  6. Save summary JSON

Usage:
    python phase0_datacentric.py \
        --checkpoint results/lmb/lmb_ablation_cb8k/checkpoints/best_model.pt \
        --data-root /path/to/imagenet \
        --num-samples 50000

    # Quick test:
    python phase0_datacentric.py \
        --checkpoint results/lmb/lmb_ablation_cb8k/checkpoints/best_model.pt \
        --num-samples 1000
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from quantization.model import ModelConfig, UnifiedAutoEncoder, QuantizerConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase 0: Data-Centric Codebook Initialisation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained LGQ checkpoint (best_model.pt)")
    parser.add_argument("--data-root", type=str,
                        default="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet",
                        help="Path to ImageNet dataset")
    parser.add_argument("--dataset", type=str, default="imagenet",
                        choices=["imagenet", "imagenet100"],
                        help="Dataset to use")
    parser.add_argument("--num-samples", type=int, default=50000,
                        help="Number of image patches to encode (M')")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for encoding")
    parser.add_argument("--image-size", type=int, default=128,
                        help="Input image size")
    parser.add_argument("--output-dir", type=str, default="phase0_results",
                        help="Output directory for results")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loading workers")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--override-K", type=int, default=None,
                        help="Override recommended K (otherwise 2*N*)")
    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """Load a trained LGQ model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    # Reconstruct model from saved config
    model_type = config.get("model_type", config.get("model", "lmb"))
    dim = config.get("dim", 128)
    embedding_dim = config.get("embedding_dim", 128)

    model_cfg = ModelConfig(
        base_ch=dim // 2,
        embedding_dim=embedding_dim,
    )

    # Build quantizer config from checkpoint config
    qconfig_kwargs = {}
    if model_type == "lmb":
        qconfig_kwargs = dict(
            num_bins=config.get("num_bins", 8192),
            lambda_peak=config.get("lambda_peak", 0.005),
            lambda_bins=config.get("lambda_bins", 0.005),
            lambda_floor=config.get("lambda_floor", 0.0),
            p_min=config.get("p_min", 0.0),
            tau=config.get("tau_start", 1.0),
            learn_widths=config.get("learn_widths", False),
            flatten_channels=config.get("flatten_channels", True),
            init_method=config.get("init_method", "random"),
            perchannel_fair=config.get("perchannel_fair", False),
            lmb_levels=config.get("lmb_levels", [16, 16, 8, 8]),
        )
    elif model_type == "vq":
        qconfig_kwargs = dict(
            codebook_size=config.get("codebook_size", 8192),
            commitment_weight=config.get("commitment_weight", 1.0),
            decay=config.get("decay", 0.8),
        )
    elif model_type == "fsq":
        qconfig_kwargs = dict(
            levels=config.get("levels", [8, 5, 5, 5]),
        )

    quantizer_config = QuantizerConfig(**qconfig_kwargs)
    model = UnifiedAutoEncoder(model_type, model_cfg, quantizer_config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_type} model from {checkpoint_path}")
    print(f"  dim={dim}, embedding_dim={embedding_dim}")
    if model_type == "lmb":
        print(f"  num_bins={config.get('num_bins')}, flatten={config.get('flatten_channels')}")

    return model, config


def get_imagenet100_synsets(data_root):
    """Return the first 100 synset folder names (sorted)."""
    train_dir = os.path.join(data_root, "train")
    all_synsets = sorted([d for d in os.listdir(train_dir)
                          if os.path.isdir(os.path.join(train_dir, d))])
    return set(all_synsets[:100])


def create_dataset(data_root, dataset_name, image_size):
    """Create ImageNet or ImageNet-100 dataset."""
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_path = os.path.join(data_root, "train")
    full_dataset = datasets.ImageFolder(root=train_path, transform=transform)

    if dataset_name == "imagenet100":
        synsets = get_imagenet100_synsets(data_root)
        indices = [i for i, (path, _) in enumerate(full_dataset.samples)
                   if os.path.basename(os.path.dirname(path)) in synsets]
        full_dataset = Subset(full_dataset, indices)
        print(f"ImageNet-100: {len(full_dataset)} training images")
    else:
        print(f"ImageNet: {len(full_dataset)} training images")

    return full_dataset


@torch.no_grad()
def encode_patches(model, dataset, num_samples, batch_size, device, num_workers=4):
    """
    STEP 1: Encode M' image patches through the frozen encoder.
    Returns latent vectors Z of shape [M', C] where C is the latent channel dim.
    """
    print(f"\n{'='*60}")
    print(f"STEP 1: Encoding {num_samples} image patches")
    print(f"{'='*60}")

    # Subsample dataset
    total = len(dataset)
    if num_samples < total:
        indices = torch.randperm(total)[:num_samples].tolist()
        subset = Subset(dataset, indices)
    else:
        subset = dataset
        num_samples = total

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    all_latents = []
    count = 0
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)
        z = model.enc(images)  # [B, C, H, W]
        B, C, H, W = z.shape

        # Apply LayerNorm like the model does
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = model.pre_vq_norm(z_flat)

        # If there's a pre_quant projection, apply it
        if model.pre_quant is not None:
            z_2d = z_norm.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_proj = model.pre_quant(z_2d)  # [B, C', H, W]
            C_out = z_proj.shape[1]
            z_norm = z_proj.permute(0, 2, 3, 1).contiguous().view(B * H * W, C_out)

        all_latents.append(z_norm.cpu())
        count += B
        if (batch_idx + 1) % 50 == 0:
            print(f"  Encoded {count}/{num_samples} images ({B*H*W} patches/image)")

    Z = torch.cat(all_latents, dim=0)
    # Subsample patches if we have too many (keep M' total patches)
    total_patches = Z.shape[0]
    if total_patches > num_samples * 64:  # limit memory
        idx = torch.randperm(total_patches)[:num_samples * 16]
        Z = Z[idx]

    print(f"  Total latent vectors: {Z.shape[0]} x {Z.shape[1]}")
    return Z


def estimate_covariance(Z):
    """
    STEP 2: Estimate latent covariance and compute d_eff.
    """
    print(f"\n{'='*60}")
    print(f"STEP 2: Estimating covariance and effective dimension")
    print(f"{'='*60}")

    M, C = Z.shape
    Z_centered = Z - Z.mean(dim=0, keepdim=True)

    # Use SVD for numerical stability (especially if C is large)
    if C > 256:
        print(f"  Using SVD (C={C} > 256)")
        # Low-rank: Z_centered = U @ diag(S) @ V^T
        # Covariance eigenvalues = S^2 / M
        U, S, Vt = torch.linalg.svd(Z_centered, full_matrices=False)
        eigenvalues = (S ** 2) / M
        eigenvectors = Vt  # rows are eigenvectors
    else:
        print(f"  Computing full covariance (C={C})")
        Sigma = (Z_centered.T @ Z_centered) / M
        eigenvalues, eigenvectors = torch.linalg.eigh(Sigma)
        # Sort descending
        idx = eigenvalues.argsort(descending=True)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

    # Ensure non-negative
    eigenvalues = eigenvalues.clamp(min=0)

    # Effective intrinsic dimension (participation ratio)
    sum_lambda = eigenvalues.sum()
    sum_lambda_sq = (eigenvalues ** 2).sum()
    d_eff = float((sum_lambda ** 2) / sum_lambda_sq)

    print(f"  Top 5 eigenvalues: {eigenvalues[:5].tolist()}")
    print(f"  Effective dimension d_eff = {d_eff:.2f}")
    print(f"  Latent dimension C = {C}")

    return eigenvalues, eigenvectors, d_eff


def save_eigenvalue_plot(eigenvalues, output_dir):
    """Save eigenvalue spectrum plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(range(1, len(eigenvalues) + 1), eigenvalues.numpy(), "b-o", markersize=3)
    ax.set_xlabel("Component index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Latent Covariance Eigenvalue Spectrum")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "eigenvalue_spectrum.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved eigenvalue spectrum plot")


def predict_optimal_codebook_size(Z, eigenvalues, d_eff, C):
    """
    STEP 3: Predict optimal codebook size N* via Gersho-Zador formula.
    """
    print(f"\n{'='*60}")
    print(f"STEP 3: Predicting optimal codebook size N*")
    print(f"{'='*60}")

    # Target distortion D*: 10th percentile of squared pairwise distances / C
    M = Z.shape[0]
    n_pairs = min(100000, M * (M - 1) // 2)
    idx1 = torch.randint(0, M, (n_pairs,))
    idx2 = torch.randint(0, M, (n_pairs,))
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    dists_sq = ((Z[idx1] - Z[idx2]) ** 2).sum(dim=1)
    D_star = float(torch.quantile(dists_sq, 0.10)) / C
    print(f"  Target distortion D* = {D_star:.6f}")

    d = d_eff
    floor_eig = max(1e-10, float(eigenvalues.max()) * 1e-4)
    pos_eigs = eigenvalues[eigenvalues > floor_eig]
    log_det_Sigma = float(pos_eigs.log().sum())
    print(f"  Using {len(pos_eigs)}/{len(eigenvalues)} eigenvalues (floor={floor_eig:.3e}) for norm factor")
    print(f"  log det Sigma = {log_det_Sigma:.4f}")

    # ||p_Z||_{d/(d+2)} ~ (2*pi)^{-d/(d+2)} * exp(-log_det / (d+2))
    # Work in log space to avoid overflow
    log_norm_factor = -d / (d + 2) * np.log(2 * np.pi) - log_det_Sigma / (d + 2)
    print(f"  log(norm_factor) = {log_norm_factor:.4f}")

    # Gersho-Zador: N* >= [C_d * norm_factor / D*]^{d/2}
    # C_d = 1/12 (approximate Zador constant)
    C_d = 1.0 / 12.0
    # log(N*) = (d/2) * [log(C_d) + log_norm_factor - log(D*)]
    log_N_star = (d / 2) * (np.log(C_d) + log_norm_factor - np.log(max(D_star, 1e-20)))
    print(f"  log(N*) = {log_N_star:.4f}")

    # Clamp to reasonable range for VQ codebooks
    log_N_star_clamped = np.clip(log_N_star, 0, np.log(131072))
    if log_N_star != log_N_star_clamped:
        print(f"  NOTE: log(N*) = {log_N_star:.2f} clamped to [{0:.1f}, {np.log(131072):.1f}]")
    N_star_raw = np.exp(log_N_star)
    N_star = max(1, min(int(np.ceil(N_star_raw)), 131072))  # cap at 128K
    recommended_K = 2 * N_star  # safety margin

    print(f"  N* (raw) = {N_star_raw:.1f}")
    print(f"  N* = {N_star}")
    print(f"  Recommended K = 2 * N* = {recommended_K}")

    return N_star, recommended_K, D_star


def density_weighted_kmeans(Z, K, eigenvalues, d_eff, seed=42):
    """
    STEP 4: Density-weighted k-means initialisation at Gersho-optimal density.

    Weight each sample z by:
        w(z) = exp( ||Sigma^{-1/2}(z - z_bar)||^2 / (d_eff + 2) )
    This tilts sampling from p_Z toward p_Z^{d/(d+2)} = Lambda*.
    """
    print(f"\n{'='*60}")
    print(f"STEP 4: Density-weighted k-means (K={K})")
    print(f"{'='*60}")

    from sklearn.cluster import KMeans

    M, C = Z.shape
    Z_np = Z.numpy().astype(np.float64)
    z_bar = Z_np.mean(axis=0)

    # Compute Sigma^{-1/2} via eigendecomposition
    eigs = eigenvalues.numpy().astype(np.float64)
    # Pseudoinverse: clip small eigenvalues
    eigs_clipped = np.maximum(eigs[:C], 1e-6)
    inv_sqrt_eigs = 1.0 / np.sqrt(eigs_clipped)

    # For the covariance matrix built from these eigenvalues, we need eigenvectors too
    # But we computed Z_centered, so Sigma = V @ diag(lambda) @ V^T
    # Sigma^{-1/2} = V @ diag(1/sqrt(lambda)) @ V^T
    # Mahalanobis: ||Sigma^{-1/2}(z-z_bar)||^2 = (z-z_bar)^T Sigma^{-1} (z-z_bar)

    # Recompute covariance and its inverse sqrt efficiently
    Z_centered = Z_np - z_bar
    if C > 256:
        _, S, Vt = np.linalg.svd(Z_centered, full_matrices=False)
        cov_eigs = (S ** 2) / M
        floor = max(1e-6, float(cov_eigs.max()) * 1e-4)
        cov_eigs_clipped = np.maximum(cov_eigs, floor)
        Z_rotated = Z_centered @ Vt.T
        Z_scaled = Z_rotated / np.sqrt(cov_eigs_clipped)[np.newaxis, :]
    else:
        Sigma = (Z_centered.T @ Z_centered) / M
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        floor = max(1e-6, float(eigvals.max()) * 1e-4)
        eigvals_clipped = np.maximum(eigvals, floor)
        Z_rotated = Z_centered @ eigvecs
        Z_scaled = Z_rotated / np.sqrt(eigvals_clipped)[np.newaxis, :]

    mahal_sq = np.sum(Z_scaled ** 2, axis=1)
    log_w = mahal_sq / (d_eff + 2)
    log_w = log_w - log_w.max()
    weights = np.exp(log_w)
    weights = weights / weights.sum() * M

    print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"  Weight std/mean: {weights.std()/weights.mean():.4f}")

    # Density-weighted k-means
    print(f"  Running density-weighted k-means (K={K})...")
    kmeans_dc = KMeans(n_clusters=K, random_state=seed, n_init=3, max_iter=300)
    kmeans_dc.fit(Z_np, sample_weight=weights)
    codebook_datacentric = kmeans_dc.cluster_centers_.astype(np.float32)
    print(f"  Density-weighted k-means done. Inertia: {kmeans_dc.inertia_:.2f}")

    # Standard k-means (uniform weights)
    print(f"  Running standard k-means (K={K})...")
    kmeans_std = KMeans(n_clusters=K, random_state=seed, n_init=3, max_iter=300)
    kmeans_std.fit(Z_np)
    codebook_standard = kmeans_std.cluster_centers_.astype(np.float32)
    print(f"  Standard k-means done. Inertia: {kmeans_std.inertia_:.2f}")

    # Gaussian random init
    print(f"  Generating Gaussian random codebook (K={K})...")
    rng = np.random.RandomState(seed)
    z_mean = Z_np.mean(axis=0)
    z_std = Z_np.std(axis=0)
    codebook_gaussian = (rng.randn(K, C) * z_std + z_mean).astype(np.float32)

    return codebook_datacentric, codebook_standard, codebook_gaussian


def utilisation_analysis(Z, codebooks, names):
    """
    STEP 5: Predicted utilisation analysis.
    For each codebook, assign all M' latent vectors to nearest codeword.
    """
    print(f"\n{'='*60}")
    print(f"STEP 5: Predicted utilisation analysis")
    print(f"{'='*60}")

    results = {}
    Z_np = Z.numpy()

    for name, cb in zip(names, codebooks):
        K = cb.shape[0]
        # Compute distances in batches to avoid OOM
        batch_size = 5000
        assignments = []
        total_dist = 0.0
        for i in range(0, len(Z_np), batch_size):
            z_batch = Z_np[i:i+batch_size]
            # [batch, K]
            dists = np.sum((z_batch[:, np.newaxis, :] - cb[np.newaxis, :, :]) ** 2, axis=2)
            nearest = np.argmin(dists, axis=1)
            assignments.append(nearest)
            total_dist += np.min(dists, axis=1).sum()

        assignments = np.concatenate(assignments)
        avg_dist = total_dist / len(assignments)

        # Active codes
        unique_codes = np.unique(assignments)
        fraction_active = len(unique_codes) / K

        # Entropy of assignment distribution
        counts = np.bincount(assignments, minlength=K).astype(np.float64)
        probs = counts / counts.sum()
        probs_nonzero = probs[probs > 0]
        entropy = -np.sum(probs_nonzero * np.log2(probs_nonzero))

        print(f"  {name}:")
        print(f"    Active codes: {len(unique_codes)}/{K} ({fraction_active*100:.1f}%)")
        print(f"    Assignment entropy: {entropy:.2f} bits (max {np.log2(K):.2f})")
        print(f"    Avg distance to nearest: {avg_dist:.6f}")

        results[name] = {
            "fraction_active": fraction_active,
            "active_codes": int(len(unique_codes)),
            "entropy_bits": float(entropy),
            "avg_distance": float(avg_dist),
        }

    return results


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Disable cuDNN to avoid segfaults on A100s with system torch 2.2.0+cu118
    torch.backends.cudnn.enabled = False

    # Create output directory
    output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    model, config = load_model_from_checkpoint(args.checkpoint, device)

    # Create dataset
    dataset = create_dataset(args.data_root, args.dataset, args.image_size)

    # STEP 1: Encode patches
    Z = encode_patches(model, dataset, args.num_samples, args.batch_size,
                       device, args.num_workers)

    # Move to CPU for covariance computation
    Z = Z.float()

    # STEP 2: Covariance and d_eff
    eigenvalues, eigenvectors, d_eff = estimate_covariance(Z)
    save_eigenvalue_plot(eigenvalues, output_dir)

    # STEP 3: Predict N*
    C = Z.shape[1]
    N_star, recommended_K, D_star = predict_optimal_codebook_size(
        Z, eigenvalues, d_eff, C
    )

    # Override K if specified
    K = args.override_K if args.override_K is not None else recommended_K
    # K cannot exceed the number of samples
    K = min(K, Z.shape[0] - 1)

    # STEP 4: Density-weighted k-means
    codebook_dc, codebook_std, codebook_gauss = density_weighted_kmeans(
        Z, K, eigenvalues, d_eff, seed=args.seed
    )

    # Save codebooks
    np.save(os.path.join(output_dir, "codebook_datacentric.npy"), codebook_dc)
    np.save(os.path.join(output_dir, "codebook_standard_kmeans.npy"), codebook_std)
    np.save(os.path.join(output_dir, "codebook_gaussian_init.npy"), codebook_gauss)
    print(f"\n  Saved codebooks to {output_dir}/")

    # STEP 5: Utilisation analysis
    names = ["datacentric", "standard_kmeans", "gaussian"]
    codebooks = [codebook_dc, codebook_std, codebook_gauss]
    util_results = utilisation_analysis(Z, codebooks, names)

    # STEP 6: Save summary JSON
    summary = {
        "d_eff": float(d_eff),
        "N_star": int(N_star),
        "recommended_K": int(recommended_K),
        "K_used": int(K),
        "D_star": float(D_star),
        "latent_dim_C": int(C),
        "num_samples": int(Z.shape[0]),
        "eigenvalue_spectrum": eigenvalues[:50].tolist(),
        "predicted_utilisation": {
            name: util_results[name]["fraction_active"]
            for name in names
        },
        "predicted_avg_distance": {
            name: util_results[name]["avg_distance"]
            for name in names
        },
        "predicted_entropy_bits": {
            name: util_results[name]["entropy_bits"]
            for name in names
        },
        "predicted_active_codes": {
            name: util_results[name]["active_codes"]
            for name in names
        },
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"PHASE 0 COMPLETE")
    print(f"{'='*60}")
    print(f"  d_eff = {d_eff:.2f}")
    print(f"  N* = {N_star}")
    print(f"  Recommended K = {recommended_K}")
    print(f"  D* = {D_star:.6f}")
    print(f"  Summary saved to {summary_path}")
    print(f"  Codebooks saved to {output_dir}/")


if __name__ == "__main__":
    main()

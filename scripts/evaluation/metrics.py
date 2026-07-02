#!/usr/bin/env python3
"""
Image quality metrics for evaluating reconstruction quality.

Metrics:
- PSNR (Peak Signal-to-Noise Ratio) ↑ - higher is better
- SSIM (Structural Similarity Index) ↑ - higher is better  
- LPIPS (Learned Perceptual Image Patch Similarity) ↓ - lower is better
- Codebook Utilization ↑ - percentage of codebook used
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class ImageMetrics:
    """Container for image quality metrics."""
    psnr: float
    ssim: float
    lpips: Optional[float] = None


def compute_psnr(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        max_val: Maximum pixel value (1.0 for normalized, 255.0 for uint8)
    
    Returns:
        PSNR in dB (higher is better)
    """
    mse = F.mse_loss(reconstructed, original, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(max_val ** 2 / mse)
    return float(psnr.item())


def compute_ssim(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    window_size: int = 11,
    data_range: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        window_size: Size of the Gaussian window
        data_range: Dynamic range of pixel values
    
    Returns:
        SSIM score (higher is better, max 1.0)
    """
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.view(1, 1, -1) * g.view(1, 1, -1).transpose(-1, -2)
    
    window = gaussian_window(window_size).to(original.device)
    window = window.expand(original.size(1), 1, window_size, window_size)
    
    # Compute means
    mu1 = F.conv2d(original, window, padding=window_size // 2, groups=original.size(1))
    mu2 = F.conv2d(reconstructed, window, padding=window_size // 2, groups=original.size(1))
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = F.conv2d(original ** 2, window, padding=window_size // 2, groups=original.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(reconstructed ** 2, window, padding=window_size // 2, groups=original.size(1)) - mu2_sq
    sigma12 = F.conv2d(original * reconstructed, window, padding=window_size // 2, groups=original.size(1)) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim_map.mean().item())


class LPIPSMetric:
    """
    LPIPS (Learned Perceptual Image Patch Similarity) metric.
    
    Uses a pretrained VGG network to compute perceptual similarity.
    Lower values indicate more similar images.
    """
    
    def __init__(self, device: str = "cuda", net: str = "vgg"):
        self.device = device
        self._model = None
        self._net = net
    
    @property
    def model(self):
        """Lazily load LPIPS model."""
        if self._model is None:
            try:
                import lpips
                print("Loading LPIPS model...")
                self._model = lpips.LPIPS(net=self._net).to(self.device)
                self._model.eval()
            except ImportError:
                raise ImportError(
                    "LPIPS requires the lpips package. Install with: pip install lpips"
                )
        return self._model
    
    @torch.no_grad()
    def compute(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
    ) -> float:
        """
        Compute LPIPS between original and reconstructed images.
        
        Callers should pass denormalized images in [0, 1]. We convert both to
        [-1, 1] explicitly and call the library with normalize=False so both
        inputs are treated identically (avoids library quirks).
        
        Args:
            original: Original images [B, C, H, W], values in [-1, 1] or [0, 1]
            reconstructed: Reconstructed images [B, C, H, W]
        
        Returns:
            LPIPS score (lower is better), mean over this batch
        """
        original = original.float().to(self.device)
        reconstructed = reconstructed.float().to(self.device)
        if not original.is_contiguous():
            original = original.contiguous()
        if not reconstructed.is_contiguous():
            reconstructed = reconstructed.contiguous()

        # Reject invalid reconstructions so we don't pollute the metric
        if not torch.isfinite(reconstructed).all() or not torch.isfinite(original).all():
            return float("nan")

        # LPIPS expects [-1, 1]. Convert [0, 1] -> [-1, 1] for BOTH inputs
        # so they are treated identically (some lib versions only normalize in0)
        if original.min() >= 0 and original.max() <= 1:
            original = original * 2.0 - 1.0
            reconstructed = reconstructed * 2.0 - 1.0
        elif reconstructed.min() >= 0 and reconstructed.max() <= 1:
            reconstructed = reconstructed * 2.0 - 1.0

        try:
            lpips_val = self.model(original, reconstructed, normalize=False)
        except TypeError:
            lpips_val = self.model(original, reconstructed)
        out = lpips_val.mean()
        return float(out.item())


def compute_codebook_utilization(
    indices: torch.Tensor,
    codebook_size: int,
) -> Dict[str, float]:
    """
    Compute codebook utilization metrics.
    
    Args:
        indices: Quantization indices [N] or any shape (will be flattened)
        codebook_size: Total size of the codebook
    
    Returns:
        Dict with:
            - active_codes: Number of unique codes used
            - utilization: Percentage of codebook used (0-100)
            - perplexity: Exponential of entropy (effective codebook size)
            - entropy_bits: Empirical marginal entropy in bits (log2(perplexity))
            - bits_per_token: Bits per token under i.i.d. marginal (same as entropy_bits)
    """
    indices = indices.view(-1).long()
    
    # Count unique codes
    unique_codes = indices.unique()
    active_codes = int(unique_codes.numel())
    
    # Compute utilization percentage
    utilization = 100.0 * active_codes / codebook_size
    
    # Compute perplexity and empirical entropy (marginal) in bits
    counts = torch.bincount(indices, minlength=codebook_size)
    counts = counts[counts > 0].float()
    probs = counts / counts.sum()
    entropy_nats = -(probs * probs.log()).sum()
    perplexity = float(torch.exp(entropy_nats).item())
    # H in bits = log2(perplexity); bits-per-token under i.i.d. marginal = same
    entropy_bits = float((entropy_nats / math.log(2)).item()) if perplexity > 0 else 0.0
    bits_per_token = entropy_bits

    return {
        "active_codes": active_codes,
        "codebook_utilization": utilization,
        "perplexity": perplexity,
        "entropy_bits": entropy_bits,
        "bits_per_token": bits_per_token,
    }


class MetricsCalculator:
    """
    Unified metrics calculator for image reconstruction evaluation.
    
    Computes: PSNR, SSIM, LPIPS, Codebook Utilization, rFID
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self._lpips = None
        self._fid_calculator = None
    
    @property
    def lpips(self) -> LPIPSMetric:
        """Lazily initialize LPIPS."""
        if self._lpips is None:
            self._lpips = LPIPSMetric(device=self.device)
        return self._lpips
    
    @property
    def fid_calculator(self):
        """Lazily initialize FID calculator."""
        if self._fid_calculator is None:
            from scripts.evaluation.fid import FIDCalculator
            self._fid_calculator = FIDCalculator(device=self.device)
        return self._fid_calculator
    
    @torch.no_grad()
    def compute_batch_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        compute_lpips: bool = True,
    ) -> Dict[str, float]:
        """
        Compute image quality metrics for a batch.
        
        Args:
            original: Original images [B, C, H, W]
            reconstructed: Reconstructed images [B, C, H, W]
            compute_lpips: Whether to compute LPIPS (slower)
        
        Returns:
            Dict with psnr, ssim, and optionally lpips
        """
        metrics = {}
        
        # PSNR
        metrics["psnr"] = compute_psnr(original, reconstructed)
        
        # SSIM
        metrics["ssim"] = compute_ssim(original, reconstructed)
        
        # LPIPS
        if compute_lpips:
            try:
                metrics["lpips"] = self.lpips.compute(original, reconstructed)
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")
                metrics["lpips"] = float("nan")
        
        return metrics
    
    def compute_codebook_metrics(
        self,
        indices: torch.Tensor,
        codebook_size: int,
    ) -> Dict[str, float]:
        """Compute codebook utilization metrics."""
        return compute_codebook_utilization(indices, codebook_size)

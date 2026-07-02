#!/usr/bin/env python3
"""
FID (Fréchet Inception Distance) calculation utilities.

rFID measures how similar the distribution of reconstructed images is to
the original images using features from an Inception v3 network.

Uses pytorch-fid library when available (standard, reproducible FID).
Falls back to hand-implemented version if pytorch-fid is not installed.

Lower FID = better reconstruction quality.

Inception normalization convention (matches pytorch-fid):
  - Input images must be in [0, 1] range before being passed here.
  - The InceptionV3Features module internally rescales to [-1, 1] as expected
    by the Inception v3 network weights.
"""

import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import numpy as np
from typing import Optional, Tuple
try:
    from tqdm.auto import tqdm as _tqdm
    def _wrap_progress(it, **kwargs):
        return _tqdm(it, **kwargs)
except ImportError:
    def _wrap_progress(it, **kwargs):
        return it

# Prefer pytorch-fid library when available (standard, reproducible FID)
PYTORCH_FID_AVAILABLE = False
try:
    from pytorch_fid import fid_score as _pytorch_fid_score
    PYTORCH_FID_AVAILABLE = True
except ImportError:
    import warnings
    warnings.warn(
        "pytorch-fid not installed; using hand-implemented FID. "
        "Install with: pip install pytorch-fid"
    )


class InceptionV3Features(nn.Module):
    """
    Inception v3 model that extracts 2048-dim pool features for FID.

    Follows the same convention as pytorch-fid:
      - Input: images in [0, 1] (float32, shape [B, 3, H, W])
      - Internally rescaled to [-1, 1] before the Inception forward pass
      - Output: [B, 2048] feature vectors from the final avg-pool layer

    Using aux_logits=False avoids the auxiliary branch that is not needed
    for feature extraction and makes the forward pass a clean sequential
    path through the main trunk.
    """

    def __init__(self, device: str = "cuda"):
        super().__init__()

        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.IMAGENET1K_V1,
            aux_logits=True,
        )
        inception.aux_logits = False  # disable aux branch at runtime
        inception.AuxLogits = None
        inception.eval()

        # Remove the final FC classifier; keep everything up to and including
        # the adaptive avg pool. We expose the pool output via a forward hook.
        self._pool_features: Optional[torch.Tensor] = None

        def _hook(module, input, output):
            # output: [B, 2048, 1, 1]
            self._pool_features = output.view(output.size(0), -1)

        inception.avgpool.register_forward_hook(_hook)

        # Wrap the full inception model (minus the fc head) so we can call
        # forward() and read pool features from the hook.
        inception.fc = nn.Identity()  # disable fc
        self._inception = inception

        self.to(device)
        self.device = device

        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract 2048-dimensional features from images.

        Args:
            x: Images in [0, 1], shape [B, 3, H, W]

        Returns:
            Feature tensor [B, 2048]
        """
        # Resize to Inception input size (299x299) if needed
        if x.shape[-2] != 299 or x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)

        # Rescale [0, 1] → [-1, 1] as Inception v3 weights expect
        x = x * 2.0 - 1.0

        self._inception(x)  # forward fills self._pool_features via hook
        return self._pool_features


def compute_statistics(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and covariance of features.

    Args:
        features: Feature matrix [N, D]

    Returns:
        Tuple of (mean [D], covariance [D, D])
    """
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mu, sigma


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Compute Fréchet distance between two Gaussians.

    FID = ||mu1 - mu2||^2 + Tr(sigma1 + sigma2 - 2*sqrt(sigma1 @ sigma2))

    Args:
        mu1, sigma1: Mean and covariance of first distribution
        mu2, sigma2: Mean and covariance of second distribution
        eps: Small constant for numerical stability

    Returns:
        FID score (lower is better)
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2

    # Product of covariance matrices can be near-singular in Inception feature
    # space (2048-dim, limited samples), which makes `sqrtm` return a matrix
    # with a noticeable imaginary component. If the naive sqrtm fails the 1e-3
    # imaginary tolerance, progressively jitter both sigmas and retry rather
    # than raising (which produces rFID=NaN in eval_metrics).
    def _sqrtm(s1: np.ndarray, s2: np.ndarray):
        return linalg.sqrtm(s1.dot(s2), disp=False)[0]

    covmean = _sqrtm(sigma1, sigma2)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = _sqrtm(sigma1 + offset, sigma2 + offset)

    if np.iscomplexobj(covmean):
        for jitter in (eps, 1e-4, 1e-3, 1e-2):
            if np.max(np.abs(covmean.imag)) <= 1e-3:
                break
            offset = np.eye(sigma1.shape[0]) * jitter
            covmean = _sqrtm(sigma1 + offset, sigma2 + offset)
            if not np.iscomplexobj(covmean):
                break
        if np.iscomplexobj(covmean):
            if np.max(np.abs(covmean.imag)) > 1e-3:
                # Still non-negligible imaginary part; take real part and
                # warn rather than fail so training metrics remain populated.
                import warnings
                warnings.warn(
                    f"FID sqrtm imaginary component {np.max(np.abs(covmean.imag)):.4f} "
                    "exceeds 1e-3 after jitter; using real part",
                    RuntimeWarning,
                )
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # When sigma1, sigma2 are near-singular in InceptionV3 feature space (2048
    # dim) the Schur-based sqrtm picks a wrong matrix-sign branch and the real
    # part of `covmean` overshoots the true trace, producing FID < 0. Recover
    # via the principal-branch identity tr(sqrt(s1@s2)) = sum(sqrt(eigvals)).
    if not np.isfinite(fid) or fid < 0.0:
        eigvals = np.linalg.eigvals(sigma1.dot(sigma2))
        # Eigenvalues of a product of PSD matrices are real and non-negative;
        # numerical noise can flip tiny ones. Clip to >=0 before sqrt.
        eig_real = np.clip(np.real(eigvals), 0.0, None)
        tr_covmean = float(np.sum(np.sqrt(eig_real)))
        fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
        if fid < 0.0:
            # Residual numerical underflow; clamp.
            fid = 0.0

    return float(fid)


def _compute_rfid_pytorch_fid(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    max_samples: Optional[int] = None,
    denormalize_fn=None,
    show_progress: bool = True,
    batch_size: int = 50,
) -> float:
    """
    Compute rFID using pytorch-fid library (saves images to temp dirs).
    Images are saved as PNG (lossless) so no compression artifacts.
    """
    from torchvision.utils import save_image

    model.eval()
    if denormalize_fn is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        denormalize_fn = lambda x: (x * std + mean).clamp(0, 1)  # noqa: E731

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_dir = os.path.join(tmpdir, "orig")
        recon_dir = os.path.join(tmpdir, "recon")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(recon_dir, exist_ok=True)

        idx = 0
        iterator = (
            _wrap_progress(dataloader, desc="Saving images for pytorch-fid rFID", leave=False)
            if show_progress
            else dataloader
        )
        with torch.no_grad():
            for batch in iterator:
                if isinstance(batch, (tuple, list)):
                    images = batch[0]
                else:
                    images = batch
                images = images.to(device)
                outputs = model(images)
                recon = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
                orig_denorm = denormalize_fn(images)
                recon_denorm = denormalize_fn(recon)
                for i in range(images.size(0)):
                    save_image(
                        orig_denorm[i : i + 1],
                        os.path.join(orig_dir, f"{idx:06d}.png"),
                    )
                    save_image(
                        recon_denorm[i : i + 1],
                        os.path.join(recon_dir, f"{idx:06d}.png"),
                    )
                    idx += 1
                if max_samples and idx >= max_samples:
                    break

        if idx == 0:
            return float("nan")

        try:
            fid_value = _pytorch_fid_score.calculate_fid_given_paths(
                [orig_dir, recon_dir],
                batch_size=batch_size,
                device=torch.device(device),
                dims=2048,
                num_workers=0,
            )
            fid_value = float(fid_value)
            # pytorch-fid takes the real part of a complex sqrtm result without
            # validating sign; for near-singular covariances this can yield
            # FID < 0. Recompute via the eigenvalue path in compute_fid which
            # clamps negative numerical noise.
            if not (fid_value >= 0.0 and np.isfinite(fid_value)):
                import warnings
                warnings.warn(
                    f"pytorch-fid returned non-physical FID={fid_value}; "
                    "recomputing via eigenvalue fallback",
                    RuntimeWarning,
                )
                return _compute_rfid_from_paths(orig_dir, recon_dir, device=device)
            return fid_value
        except ValueError as e:
            # pytorch-fid raises ValueError("Imaginary component {m}") when the
            # covariance-matrix sqrtm has non-trivial imaginary part. Reuse the
            # saved images but extract features + compute FID ourselves so the
            # jitter/real-part fallback in `compute_fid` can recover.
            import warnings
            warnings.warn(
                f"pytorch-fid failed ({e}); falling back to hand-implemented FID with jitter",
                RuntimeWarning,
            )
            return _compute_rfid_from_paths(orig_dir, recon_dir, device=device)


def _compute_rfid_from_paths(orig_dir: str, recon_dir: str, device: str) -> float:
    """Fallback rFID: extract InceptionV3 features from two image directories
    using pytorch-fid's own feature extractor (so statistics match exactly),
    then run our robust `compute_fid` with jitter.
    """
    from pytorch_fid.inception import InceptionV3
    from pytorch_fid.fid_score import compute_statistics_of_path

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()

    mu1, sigma1 = compute_statistics_of_path(orig_dir, inception, 50, 2048, torch.device(device), 0)
    mu2, sigma2 = compute_statistics_of_path(recon_dir, inception, 50, 2048, torch.device(device), 0)
    return compute_fid(mu1, sigma1, mu2, sigma2)


@torch.no_grad()
def compute_rfid(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    inception_model: Optional[InceptionV3Features] = None,
    max_samples: Optional[int] = None,
    denormalize_fn=None,
    show_progress: bool = True,
    use_library: bool = True,
) -> float:
    """
    Compute rFID (reconstruction FID) for a model on a dataset.

    Compares the distribution of original images to reconstructed images.
    Uses pytorch-fid library when available and use_library=True.

    Args:
        model: Autoencoder model that returns (reconstruction, indices, ...)
        dataloader: DataLoader with images
        device: Device to use
        inception_model: Pretrained InceptionV3Features (used only in fallback path)
        max_samples: Maximum number of samples to use
        denormalize_fn: Function to map training-normalized images back to [0, 1]
        show_progress: Show progress bar
        use_library: If True and pytorch-fid available, use it; else hand-implemented

    Returns:
        rFID score (lower is better)
    """
    if use_library and PYTORCH_FID_AVAILABLE:
        return _compute_rfid_pytorch_fid(
            model=model,
            dataloader=dataloader,
            device=device,
            max_samples=max_samples,
            denormalize_fn=denormalize_fn,
            show_progress=show_progress,
        )

    # --- Hand-implemented fallback ---
    model.eval()

    if inception_model is None:
        inception_model = InceptionV3Features(device=device)
    inception_model.eval()

    # Default denormalization for ImageNet-normalized inputs
    if denormalize_fn is None:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        denormalize_fn = lambda x: (x * std + mean).clamp(0, 1)  # noqa: E731

    original_features = []
    recon_features = []
    samples_collected = 0

    iterator = (
        _wrap_progress(dataloader, desc="Computing rFID", leave=False)
        if show_progress
        else dataloader
    )
    for batch in iterator:
        if isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch

        images = images.to(device)

        outputs = model(images)
        recon = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        # Denormalize to [0, 1] before passing to InceptionV3Features
        images_denorm = denormalize_fn(images)
        recon_denorm = denormalize_fn(recon)

        orig_feat = inception_model(images_denorm).cpu().numpy()
        recon_feat = inception_model(recon_denorm).cpu().numpy()

        original_features.append(orig_feat)
        recon_features.append(recon_feat)

        samples_collected += images.size(0)
        if max_samples and samples_collected >= max_samples:
            break

    original_features = np.concatenate(original_features, axis=0)
    recon_features = np.concatenate(recon_features, axis=0)

    if max_samples:
        original_features = original_features[:max_samples]
        recon_features = recon_features[:max_samples]

    mu1, sigma1 = compute_statistics(original_features)
    mu2, sigma2 = compute_statistics(recon_features)

    return compute_fid(mu1, sigma1, mu2, sigma2)


class FIDCalculator:
    """
    Reusable FID calculator that caches the Inception model.

    The Inception model is only needed for the hand-implemented fallback path.
    When pytorch-fid is available it is used directly and the cached model is
    not exercised, but it is kept so the same object works regardless of backend.

    Usage:
        fid_calc = FIDCalculator(device="cuda")
        rfid = fid_calc.compute_rfid(model, dataloader)
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._inception_model: Optional[InceptionV3Features] = None
        self._backend_logged = False

    @property
    def inception_model(self) -> InceptionV3Features:
        """Lazily load Inception model (used by the hand-implemented fallback)."""
        if self._inception_model is None:
            print("Loading Inception v3 model for FID calculation...")
            self._inception_model = InceptionV3Features(device=self.device)
        return self._inception_model

    def compute_rfid(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> float:
        """Compute rFID for a model. Uses pytorch-fid library when available."""
        backend = "pytorch-fid" if PYTORCH_FID_AVAILABLE else "hand-implemented"
        if not self._backend_logged:
            print(f"rFID backend: {backend} (max_samples={max_samples})")
            self._backend_logged = True
        return compute_rfid(
            model=model,
            dataloader=dataloader,
            device=self.device,
            # Pass the cached model; it is used only in the fallback path.
            inception_model=self.inception_model if not PYTORCH_FID_AVAILABLE else None,
            max_samples=max_samples,
            show_progress=show_progress,
        )

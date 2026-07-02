"""Perceptual loss using LPIPS (VGG features) for training."""

import torch
import torch.nn as nn


class PerceptualLoss(nn.Module):
    """LPIPS-based perceptual loss for training (keeps gradients, unlike eval metric)."""

    def __init__(self, device="cuda", net="vgg"):
        super().__init__()
        import lpips
        self.model = lpips.LPIPS(net=net).to(device)
        self.model.eval()
        # Freeze VGG weights
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.

        Args:
            x: Original images [B, C, H, W] in [0, 1]
            recon: Reconstructed images [B, C, H, W] in [0, 1]

        Returns:
            Scalar perceptual loss (mean over batch).
        """
        # LPIPS expects [-1, 1]
        x_lpips = x * 2.0 - 1.0
        recon_lpips = recon * 2.0 - 1.0
        return self.model(x_lpips, recon_lpips, normalize=False).mean()

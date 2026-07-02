"""
SoftVQ quantizer (Chen et al., CVPR 2025).

Soft posteriors via temperature-scaled softmax over codebook dot products.
Produces continuous tokens (weighted sum) during training; discrete at inference.
Drop-in replacement that shares our encoder/decoder backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftVectorQuantizer(nn.Module):
    """SoftVQ-VAE quantizer: soft assignment via temperature-scaled softmax."""

    def __init__(
        self,
        codebook_size: int = 8192,
        embedding_dim: int = 64,
        tau: float = 0.07,
        entropy_loss_ratio: float = 0.01,
        l2_norm: bool = True,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.tau = tau
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, C, H, W] encoder output (C == embedding_dim)
        Returns:
            z_q: [B, C, H, W] quantized (soft weighted sum)
            entropy_loss: scalar
            indices: [B, H*W] hard argmax indices (for eval/logging)
        """
        B, C, H, W = z.shape
        # Reshape to [B, H*W, C]
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

        if self.l2_norm:
            z_flat = F.normalize(z_flat, dim=-1)
            emb = F.normalize(self.embedding.weight, dim=-1)
        else:
            emb = self.embedding.weight

        # Dot-product logits: [B, H*W, K]
        logits = torch.einsum("bnd,kd->bnk", z_flat, emb)

        # Soft assignment
        probs = F.softmax(logits / self.tau, dim=-1)  # [B, H*W, K]

        # Soft quantized output (weighted sum)
        z_q = torch.einsum("bnk,kd->bnd", probs, emb)  # [B, H*W, C]

        # Hard indices for logging
        indices = probs.argmax(dim=-1)  # [B, H*W]

        # Entropy loss: encourage sharp per-sample assignment + uniform batch usage
        # Per-sample entropy (should be low = sharp)
        per_sample_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        # Batch-average entropy (should be high = uniform)
        avg_probs = probs.mean(dim=(0, 1))  # [K]
        avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        entropy_loss = self.entropy_loss_ratio * (per_sample_entropy - avg_entropy)

        # Reshape back to [B, C, H, W]
        z_q = z_q.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return z_q, entropy_loss, indices

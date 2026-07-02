"""
BSQ quantizer — Binary Spherical Quantization (Zhao et al., ICLR 2025).

Quantizes each dimension to {-1, +1} via sign() with STE. Implicit codebook
of size 2^D (all binary vertices of the hypercube). Entropy computed in groups
for tractability.
Drop-in replacement that shares our encoder/decoder backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinarySphericalQuantizer(nn.Module):
    """BSQ: sign quantization on hypersphere with grouped entropy."""

    def __init__(
        self,
        embedding_dim: int = 64,
        beta: float = 0.25,
        gamma0: float = 1.0,
        gamma: float = 1.0,
        zeta: float = 0.1,
        group_size: int = 9,
        l2_norm: bool = True,
        inv_temperature: float = 1.0,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.gamma0 = gamma0
        self.gamma = gamma
        self.zeta = zeta
        self.group_size = group_size
        self.l2_norm = l2_norm
        self.inv_temperature = inv_temperature

        # Implicit codebook size per group
        self.codebook_size = 2 ** embedding_dim
        self.n_groups = (embedding_dim + group_size - 1) // group_size

        # Pre-compute sub-codebook vertices for each group
        # Each group has 2^group_size binary vertices
        gs = min(group_size, embedding_dim)
        n_sub = 2 ** gs
        bits = torch.arange(n_sub).unsqueeze(1)  # [n_sub, 1]
        positions = torch.arange(gs).unsqueeze(0)  # [1, gs]
        # Binary representation: {0, 1} -> {-1, +1}
        sub_codebook = ((bits >> positions) & 1).float() * 2 - 1  # [n_sub, gs]
        self.register_buffer("sub_codebook", sub_codebook)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, C, H, W] encoder output (C == embedding_dim)
        Returns:
            z_q: [B, C, H, W] binary quantized output
            loss: scalar (commit_loss + entropy penalty)
            indices: [B, H*W] codebook indices (binary encoding)
        """
        B, C, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, C)  # [N, C]

        if self.l2_norm:
            z_flat = F.normalize(z_flat, dim=-1)

        # Binary quantization: sign() with STE
        z_hard = torch.where(z_flat > 0, torch.ones_like(z_flat), -torch.ones_like(z_flat))
        z_q_flat = z_flat + (z_hard - z_flat).detach()  # STE

        # Commitment loss
        commit_loss = (z_flat - z_hard.detach()).pow(2).mean()

        # Grouped entropy computation
        per_sample_entropy = torch.tensor(0.0, device=z.device)
        codebook_entropy = torch.tensor(0.0, device=z.device)

        for g in range(self.n_groups):
            start = g * self.group_size
            end = min(start + self.group_size, C)
            gs = end - start
            z_g = z_flat[:, start:end]  # [N, gs]

            # Sub-codebook for this group
            sub_cb = self.sub_codebook[:2**gs, :gs]  # [2^gs, gs]

            # Soft distances: negative L2 squared → softmax
            # [N, 2^gs]
            dists = -torch.cdist(z_g.unsqueeze(0), sub_cb.unsqueeze(0), p=2).squeeze(0).pow(2)
            probs = F.softmax(dists * self.inv_temperature, dim=-1)

            # Per-sample entropy (should be low = sharp)
            ps_ent = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
            per_sample_entropy = per_sample_entropy + ps_ent

            # Codebook entropy (should be high = uniform)
            avg_probs = probs.mean(dim=0)
            cb_ent = -(avg_probs * (avg_probs + 1e-10).log()).sum()
            codebook_entropy = codebook_entropy + cb_ent

        per_sample_entropy = per_sample_entropy / self.n_groups
        codebook_entropy = codebook_entropy / self.n_groups

        entropy_penalty = (self.gamma0 * per_sample_entropy - self.gamma * codebook_entropy)
        loss = self.beta * commit_loss + self.zeta * entropy_penalty / self.inv_temperature

        # Compute indices from binary encoding
        binary_bits = ((z_hard + 1) / 2).long()  # {-1,+1} -> {0,1}
        # Only use first 20 bits to avoid overflow (2^20 = 1M effective codebook)
        max_bits = min(C, 20)
        powers = (2 ** torch.arange(max_bits, device=z.device)).long()
        indices = (binary_bits[:, :max_bits] * powers).sum(dim=-1)  # [N]
        indices = indices.view(B, H * W)

        # Reshape back
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return z_q, loss, indices

"""
IBQ quantizer — Index Backpropagation Quantization (Shi et al., ICCV 2025).

STE on full categorical distribution: computes dot-product logits over all
codebook entries, softmax → hard argmax with straight-through, so gradients
reach every codebook entry proportional to similarity.
Drop-in replacement that shares our encoder/decoder backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexPropagationQuantizer(nn.Module):
    """IBQ: dot-product softmax → hard argmax with STE."""

    def __init__(
        self,
        codebook_size: int = 8192,
        embedding_dim: int = 64,
        beta: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_temperature: float = 0.01,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_temperature = entropy_temperature

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, C, H, W] encoder output (C == embedding_dim)
        Returns:
            z_q: [B, C, H, W] quantized output (hard select, STE gradients)
            loss: scalar (quant_loss + entropy_loss)
            indices: [B, H*W] hard indices
        """
        B, C, H, W = z.shape

        # Dot-product logits: [B, K, H, W]
        logits = torch.einsum("bchw,kc->bkhw", z, self.embedding.weight)

        # Soft assignment
        soft_one_hot = F.softmax(logits, dim=1)  # [B, K, H, W]

        # Hard assignment + STE
        ind = soft_one_hot.argmax(dim=1, keepdim=True)  # [B, 1, H, W]
        hard_one_hot = torch.zeros_like(logits).scatter_(1, ind, 1.0)
        one_hot = hard_one_hot - soft_one_hot.detach() + soft_one_hot  # STE

        # Quantized output (gradients flow through soft_one_hot to all embeddings)
        z_q = torch.einsum("bkhw,kc->bchw", one_hot, self.embedding.weight)

        # Hard quantized (no gradient) for commitment loss
        z_q_hard = torch.einsum("bkhw,kc->bchw", hard_one_hot, self.embedding.weight)

        # Loss: STE reconstruction + codebook alignment + commitment
        quant_loss = (
            (z_q - z).pow(2).mean()
            + (z_q_hard.detach() - z).pow(2).mean()
            + self.beta * (z_q_hard - z.detach()).pow(2).mean()
        )

        # Entropy regularization
        flat_logits = logits.permute(0, 2, 3, 1).reshape(-1, self.codebook_size)
        scaled = flat_logits / self.entropy_temperature
        probs = F.softmax(scaled, dim=-1)
        # Per-sample entropy (high = spread)
        per_sample_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1).mean()
        # Batch-average entropy (high = uniform)
        avg_probs = probs.mean(dim=0)
        avg_entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        # Minimize per-sample, maximize batch → loss = per_sample - batch
        entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        total_loss = quant_loss + entropy_loss

        indices = ind.squeeze(1).view(B, H * W)  # [B, H*W]

        return z_q, total_loss, indices

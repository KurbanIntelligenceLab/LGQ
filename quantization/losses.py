"""
Loss definitions for LGQ (Learnable Geometric Quantization).
"""

from typing import NamedTuple
import torch


class LGQLosses(NamedTuple):
    """Loss components for LGQ training."""
    total: torch.Tensor
    recon: torch.Tensor
    vq: torch.Tensor
    perplexity: torch.Tensor


# Backward-compatible alias (the method was formerly named LMB / LMB-VAE).
LMBVAELosses = LGQLosses

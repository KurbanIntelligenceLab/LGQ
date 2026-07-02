"""LFQ model configuration."""

from argparse import ArgumentParser, Namespace
from math import log2

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class LFQConfig(BaseModelConfig):
    """LFQ (Lookup-Free Quantization) model configuration."""
    
    name = "lfq"
    
    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add LFQ-specific arguments."""
        parser.add_argument("--codebook-size", type=int, default=65536,
                            help="Number of codebook entries (must be power of 2)")
        parser.add_argument("--entropy-loss-weight", type=float, default=0.1,
                            help="Entropy loss weight")
        parser.add_argument("--diversity-gamma", type=float, default=1.0,
                            help="Diversity loss weight")
        parser.add_argument("--spherical", action="store_true",
                            help="Use spherical codebook")
        parser.add_argument("--lfq-entropy-warmup-steps", type=int, default=0,
                            help="MAGVIT-v2 entropy-weight warmup steps (default 0 = off). "
                                 "Per the LFQ paper, set to 2000 for the canonical "
                                 "3x->1x linear decay over the first 2k steps.")
        parser.add_argument("--lfq-entropy-warmup-factor", type=float, default=1.0,
                            help="MAGVIT-v2 entropy-weight starting multiplier "
                                 "(default 1.0 = off). Paper canonical = 3.0.")
    
    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create LFQ quantizer config."""
        return QuantizerConfig(
            codebook_size=args.codebook_size,
            entropy_loss_weight=args.entropy_loss_weight,
            diversity_gamma=args.diversity_gamma,
            spherical=args.spherical,
        )
    
    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        """Compute LFQ loss."""
        recon, indices, entropy_loss = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + entropy_loss
        active_codes = int(indices.unique().numel())
        
        # Compute perplexity
        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=1)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
        
        return LossOutput(
            total_loss=total_loss,
            rec_loss=rec_loss,
            aux_loss=entropy_loss,
            aux_loss_name="entropy_loss",
            active_codes=active_codes,
            perplexity=perplexity,
            recon=recon,
        )
    
    @staticmethod
    def get_model_info(args: Namespace) -> str:
        """Return LFQ-specific info."""
        quantize_dim = int(log2(args.codebook_size))
        return f"LFQ codebook size: {args.codebook_size}, Quantize dim (log2): {quantize_dim}"

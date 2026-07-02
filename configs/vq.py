"""VQ model configuration."""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class VQConfig(BaseModelConfig):
    """VQ (Vector Quantization) model configuration."""
    
    name = "vq"
    
    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add VQ-specific arguments."""
        parser.add_argument("--codebook-size", type=int, default=8192,
                            help="Number of codebook entries")
        parser.add_argument("--commitment-weight", type=float, default=1.0,
                            help="VQ commitment loss weight")
        parser.add_argument("--decay", type=float, default=0.8,
                            help="EMA decay for codebook")
    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create VQ quantizer config."""
        return QuantizerConfig(
            codebook_size=args.codebook_size,
            commitment_weight=args.commitment_weight,
            decay=args.decay,
        )
    
    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        """Compute VQ loss."""
        recon, indices, commit_loss = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + commit_loss
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
            aux_loss=commit_loss,
            aux_loss_name="commit_loss",
            active_codes=active_codes,
            perplexity=perplexity,
            recon=recon,
        )
    
    @staticmethod
    def get_model_info(args: Namespace) -> str:
        """Return VQ-specific info."""
        return f"VQ codebook size: {args.codebook_size}"

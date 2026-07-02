"""SimVQ model configuration."""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class SimVQConfig(BaseModelConfig):
    """SimVQ model configuration."""
    
    name = "sim_vq"
    
    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add SimVQ-specific arguments."""
        parser.add_argument("--codebook-size", type=int, default=8192,
                            help="Number of codebook entries")
        parser.add_argument("--use-mlp", action="store_true",
                            help="Use MLP for codebook transform")
        parser.add_argument("--commitment-weight", type=float, default=10.0,
                            help="Commitment loss weight")
    
    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create SimVQ quantizer config."""
        return QuantizerConfig(
            codebook_size=args.codebook_size,
            use_mlp=args.use_mlp,
            commitment_weight=args.commitment_weight,
        )
    
    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        """Compute SimVQ loss."""
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
        """Return SimVQ-specific info."""
        return f"SimVQ codebook size: {args.codebook_size}"

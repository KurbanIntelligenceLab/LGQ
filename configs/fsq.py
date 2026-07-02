"""FSQ model configuration."""

from argparse import ArgumentParser, Namespace
import math

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class FSQConfig(BaseModelConfig):
    """FSQ (Finite Scalar Quantization) model configuration."""
    
    name = "fsq"
    
    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add FSQ-specific arguments."""
        parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5],
                            help="FSQ quantization levels (e.g., 8 5 5 5)")
    
    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create FSQ quantizer config."""
        return QuantizerConfig(levels=args.levels)
    
    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        """Compute FSQ loss (reconstruction only, no auxiliary loss)."""
        recon, indices = model(x)
        rec_loss = (recon - x).abs().mean()
        active_codes = int(indices.unique().numel())
        
        # Compute perplexity
        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=1)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())
        
        return LossOutput(
            total_loss=rec_loss,
            rec_loss=rec_loss,
            aux_loss=torch.tensor(0.0, device=x.device),
            aux_loss_name="none",
            active_codes=active_codes,
            perplexity=perplexity,
            recon=recon,
        )
    
    @staticmethod
    def get_model_info(args: Namespace) -> str:
        """Return FSQ-specific info."""
        num_codes = math.prod(args.levels)
        return f"FSQ levels: {args.levels}, Codebook size: {num_codes}"

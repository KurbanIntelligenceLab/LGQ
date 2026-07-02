"""Rotation-trick VQ model configuration.

Uses the same setup as VQ but with the rotation trick (Fifty et al., ICLR 2025)
always enabled. This matches the quantization method from rotation_trick-main
so it can be run with our configs and backbone for the main experiment.
"""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class RotVQConfig(BaseModelConfig):
    """Rotation-trick VQ: VQ with rotation trick for gradient flow (same as rotation_trick-main method)."""

    name = "rot_vq"

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add rotation-trick VQ arguments (same as VQ)."""
        parser.add_argument("--codebook-size", type=int, default=8192,
                            help="Number of codebook entries")
        parser.add_argument("--commitment-weight", type=float, default=1.0,
                            help="VQ commitment loss weight")
        parser.add_argument("--decay", type=float, default=0.8,
                            help="EMA decay for codebook")

    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create quantizer config with rotation_trick=True."""
        return QuantizerConfig(
            codebook_size=args.codebook_size,
            commitment_weight=args.commitment_weight,
            decay=args.decay,
            rotation_trick=True,  # Always use rotation trick (rotation_trick-main method)
        )

    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        """Compute loss (same as VQ)."""
        recon, indices, commit_loss = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + commit_loss
        active_codes = int(indices.unique().numel())

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
        """Return model-specific info."""
        return f"Rotation-trick VQ codebook size: {args.codebook_size}"

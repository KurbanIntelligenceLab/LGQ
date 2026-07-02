"""SoftVQ-VAE model configuration (Chen et al., CVPR 2025)."""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class SoftVQConfig(BaseModelConfig):
    """SoftVQ-VAE: soft assignment via temperature-scaled softmax."""

    name = "softvq"

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        parser.add_argument("--codebook-size", type=int, default=8192)
        parser.add_argument("--softvq-tau", type=float, default=0.07,
                            help="Softmax temperature")
        parser.add_argument("--softvq-entropy-ratio", type=float, default=0.01,
                            help="Entropy loss weight")
        parser.add_argument("--softvq-l2-norm", action="store_true", default=True,
                            help="L2 normalize embeddings")

    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        return QuantizerConfig(
            codebook_size=args.codebook_size,
            softvq_tau=args.softvq_tau,
            softvq_entropy_ratio=args.softvq_entropy_ratio,
            softvq_l2_norm=args.softvq_l2_norm,
        )

    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        recon, indices, entropy_loss = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + entropy_loss
        active_codes = int(indices.unique().numel())

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
        return f"SoftVQ codebook size: {args.codebook_size}, tau: {args.softvq_tau}"

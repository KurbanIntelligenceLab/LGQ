"""iFSQ model configuration."""

from argparse import ArgumentParser, Namespace
import math

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class iFSQConfig(BaseModelConfig):
    """iFSQ (Improved FSQ via sigmoid-alpha bounding, arXiv:2601.17124)."""

    name = "ifsq"

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        parser.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5],
                            help="FSQ quantization levels (e.g., 8 5 5 5)")
        parser.add_argument("--ifsq-alpha", type=float, default=1.6,
                            help="Sigmoid slope; alpha=1.6 maps Gaussian -> uniform")

    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        return QuantizerConfig(levels=args.levels, ifsq_alpha=args.ifsq_alpha)

    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        recon, indices = model(x)
        rec_loss = (recon - x).abs().mean()
        active_codes = int(indices.unique().numel())

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
        num_codes = math.prod(args.levels)
        return f"iFSQ levels: {args.levels}, alpha: {args.ifsq_alpha}, codebook: {num_codes}"

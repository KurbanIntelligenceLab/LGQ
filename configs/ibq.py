"""IBQ model configuration (Shi et al., ICCV 2025)."""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class IBQConfig(BaseModelConfig):
    """IBQ: Index Backpropagation Quantization via categorical STE."""

    name = "ibq"

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        parser.add_argument("--codebook-size", type=int, default=8192)
        parser.add_argument("--ibq-beta", type=float, default=0.25,
                            help="Commitment loss weight")
        parser.add_argument("--ibq-entropy-weight", type=float, default=0.1,
                            help="Entropy regularization weight")
        parser.add_argument("--ibq-entropy-temp", type=float, default=0.01,
                            help="Temperature for entropy computation")

    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        return QuantizerConfig(
            codebook_size=args.codebook_size,
            ibq_beta=args.ibq_beta,
            ibq_entropy_weight=args.ibq_entropy_weight,
            ibq_entropy_temp=args.ibq_entropy_temp,
        )

    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        recon, indices, quant_loss = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + quant_loss
        active_codes = int(indices.unique().numel())

        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=1)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())

        return LossOutput(
            total_loss=total_loss,
            rec_loss=rec_loss,
            aux_loss=quant_loss,
            aux_loss_name="quant_loss",
            active_codes=active_codes,
            perplexity=perplexity,
            recon=recon,
        )

    @staticmethod
    def get_model_info(args: Namespace) -> str:
        return f"IBQ codebook size: {args.codebook_size}"

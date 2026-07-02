"""BSQ model configuration (Zhao et al., ICLR 2025)."""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class BSQConfig(BaseModelConfig):
    """BSQ: Binary Spherical Quantization on hypercube vertices."""

    name = "bsq"

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        parser.add_argument("--bsq-beta", type=float, default=0.25,
                            help="Commitment loss weight")
        parser.add_argument("--bsq-gamma0", type=float, default=1.0,
                            help="Per-sample entropy weight")
        parser.add_argument("--bsq-gamma", type=float, default=1.0,
                            help="Codebook entropy weight")
        parser.add_argument("--bsq-zeta", type=float, default=0.1,
                            help="Overall entropy penalty weight")
        parser.add_argument("--bsq-group-size", type=int, default=9,
                            help="Group size for entropy computation (2^g sub-codebook)")
        parser.add_argument("--bsq-l2-norm", action="store_true", default=True,
                            help="L2 normalize before quantization")
        parser.add_argument("--bsq-inv-temperature", type=float, default=1.0,
                            help="Inverse temperature for entropy softmax")

    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        return QuantizerConfig(
            bsq_beta=args.bsq_beta,
            bsq_gamma0=args.bsq_gamma0,
            bsq_gamma=args.bsq_gamma,
            bsq_zeta=args.bsq_zeta,
            bsq_group_size=args.bsq_group_size,
            bsq_l2_norm=args.bsq_l2_norm,
            bsq_inv_temperature=args.bsq_inv_temperature,
        )

    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        recon, indices, bsq_loss = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + bsq_loss
        active_codes = int(indices.unique().numel())

        flat_indices = indices.view(-1)
        counts = torch.bincount(flat_indices, minlength=1)
        counts = counts[counts > 0].float()
        probs = counts / counts.sum()
        perplexity = float(torch.exp(-(probs * probs.log()).sum()).item())

        return LossOutput(
            total_loss=total_loss,
            rec_loss=rec_loss,
            aux_loss=bsq_loss,
            aux_loss_name="bsq_loss",
            active_codes=active_codes,
            perplexity=perplexity,
            recon=recon,
        )

    @staticmethod
    def get_model_info(args: Namespace) -> str:
        return f"BSQ implicit codebook 2^{64} (grouped entropy, group_size={args.bsq_group_size})"

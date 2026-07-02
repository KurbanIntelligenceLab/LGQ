"""LGQ (Learnable Geometric Quantization) model configuration."""

from argparse import ArgumentParser, Namespace

import torch

from configs.base import BaseModelConfig, LossOutput
from quantization.model import QuantizerConfig


class LGQConfig(BaseModelConfig):
    """LGQ (Learnable Geometric Quantization) model configuration."""

    name = "lgq"

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> None:
        """Add LGQ-specific arguments."""
        parser.add_argument("--num-bins", type=int, default=128,
                            help="Number of quantization bins per channel")
        parser.add_argument("--lambda-peak", type=float, default=0.005,
                            help="Peak loss weight")
        parser.add_argument("--lambda-bins", type=float, default=0.005,
                            help="Bins loss weight")
        parser.add_argument("--lambda-floor", type=float, default=0.0,
                            help="Floor loss weight")
        parser.add_argument("--p-min", type=float, default=0.0,
                            help="Minimum probability")
        parser.add_argument("--tau-start", type=float, default=1.0,
                            help="Initial temperature")
        parser.add_argument("--tau-end", type=float, default=0.1,
                            help="Final temperature")
        parser.add_argument("--tau-schedule", type=str, default="linear",
                            choices=["linear", "cosine", "const"],
                            help="Anneal shape from tau-start to tau-end (const holds tau-start)")
        parser.add_argument("--learn-widths", action="store_true",
                            help="Learn bin widths")
        parser.add_argument("--flatten-channels", action="store_true", default=True,
                            help="Flatten channels into single vector before quantization (default: True)")
        parser.add_argument("--no-flatten-channels", action="store_false", dest="flatten_channels",
                            help="Disable flatten-channels (use per-channel scalar bins)")
        parser.add_argument("--init-method", type=str, default="random",
                            choices=["random", "uniform", "gaussian", "kmeans"],
                            help="Codebook initialization method: random (sample from data), uniform, gaussian, or kmeans")
        parser.add_argument("--distance-type", type=str, default="l2_sq",
                            choices=["l2_sq", "cosine", "dot"],
                            help="Distance function: l2_sq (squared Euclidean, paper Eq.2), cosine (bounded similarity), or dot (raw dot-product)")
        parser.add_argument("--perchannel-fair", action="store_true",
                            help="Fair per-channel: same structure as FSQ (5 channels, [8,8,8,8,4] bins)")
        parser.add_argument("--lgq-levels", "--lmb-levels", dest="lmb_levels",
                            type=int, nargs="+", default=[16, 16, 8, 8],
                            help="Per-channel bin counts for fair mode (match FSQ; e.g. 16 16 8 8 -> 16K). "
                                 "(--lmb-levels is a deprecated alias.)")

    @staticmethod
    def create_quantizer_config(args: Namespace) -> QuantizerConfig:
        """Create LGQ quantizer config."""
        perchannel_fair = getattr(args, "perchannel_fair", False)
        lmb_levels = getattr(args, "lmb_levels", [16, 16, 8, 8])
        if isinstance(lmb_levels, (int, float)):
            lmb_levels = [int(lmb_levels)]
        lmb_levels = list(lmb_levels)
        return QuantizerConfig(
            num_bins=args.num_bins,
            lambda_peak=args.lambda_peak,
            lambda_bins=args.lambda_bins,
            lambda_floor=args.lambda_floor,
            p_min=args.p_min,
            tau=args.tau_start,
            learn_widths=args.learn_widths,
            flatten_channels=getattr(args, 'flatten_channels', False),
            init_method=getattr(args, 'init_method', 'random'),
            distance_type=getattr(args, 'distance_type', 'l2_sq'),
            perchannel_fair=perchannel_fair,
            lmb_levels=lmb_levels if perchannel_fair else None,
        )
    
    @staticmethod
    def compute_loss(model, x: torch.Tensor) -> LossOutput:
        """Compute LGQ loss."""
        recon, indices, vq_loss, perplexity = model(x)
        rec_loss = (recon - x).abs().mean()
        total_loss = rec_loss + vq_loss

        inner = model.module if hasattr(model, "module") else model
        levels = getattr(inner.quantize, "levels", None)
        if levels is not None:
            # Fair per-channel: indices [B, T, C], channel c has levels[c] bins
            B, T, C = indices.shape
            max_l = max(levels)
            channel_idx = torch.arange(C, device=indices.device).view(1, 1, C).expand(B, T, C)
            unique_codes = channel_idx * max_l + indices
            active_codes = int(unique_codes.view(-1).unique().numel())
        elif inner.quantize.flatten_channels:
            active_codes = int(indices.view(-1).unique().numel())
        else:
            B, T, C = indices.shape
            num_bins = int(indices.max().item()) + 1
            channel_idx = torch.arange(C, device=indices.device).view(1, 1, C).expand(B, T, C)
            unique_codes = channel_idx * num_bins + indices
            active_codes = int(unique_codes.view(-1).unique().numel())
        
        ppl_val = float(perplexity.item()) if perplexity is not None else None
        return LossOutput(
            total_loss=total_loss,
            rec_loss=rec_loss,
            aux_loss=vq_loss,
            aux_loss_name="vq_loss",
            active_codes=active_codes,
            perplexity=ppl_val,
            recon=recon,
        )
    
    @staticmethod
    def get_model_info(args: Namespace) -> str:
        """Return LGQ-specific info."""
        if getattr(args, "perchannel_fair", False):
            lv = getattr(args, "lmb_levels", [16, 16, 8, 8])
            n = 1
            for x in lv:
                n *= int(x)
            return f"LGQ per-channel fair: levels={lv}, codebook={n} (matches FSQ)"
        return f"LGQ num_bins: {args.num_bins}"
    
    @staticmethod
    def update_model_for_epoch(model, args: Namespace, epoch: int, total_epochs: int) -> str:
        """Update model state for epoch (e.g., temperature annealing). Returns info string."""
        sched = getattr(args, "tau_schedule", "linear")
        if sched == "const" or args.tau_start == args.tau_end:
            model.tau = args.tau_start
            return f"Temperature: {args.tau_start:.4f} (const)"
        progress = (epoch - 1) / max(total_epochs - 1, 1)
        if sched == "cosine":
            import math
            frac = 0.5 * (1.0 - math.cos(math.pi * progress))  # smooth 0->1
        else:  # linear
            frac = progress
        tau = args.tau_start + (args.tau_end - args.tau_start) * frac
        model.tau = tau
        return f"Temperature: {tau:.4f} ({sched})"

"""
Unified autoencoder model for all quantization methods.

Single class that supports FSQ, VQ, LFQ, SimVQ, and LGQ quantizers,
all sharing the same Encoder/Decoder backbone for fair comparison.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from math import log2, prod
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn

from quantization.backbones import Encoder, Decoder


QuantizerType = Literal["fsq", "ifsq", "vq", "lfq", "sim_vq", "lgq", "lmb", "rot_vq", "softvq", "ibq", "bsq"]


@dataclass
class ModelConfig:
    """Configuration for unified autoencoder model."""
    in_channels: int = 3
    base_ch: int = 128              # Base channels for encoder/decoder
    embedding_dim: int = 256        # Latent dimension
    attn_res: tuple = (16,)         # Resolutions for attention
    legacy_decoder: bool = False    # Use old decoder (ch = max(32, emb_dim))


@dataclass  
class QuantizerConfig:
    """Configuration for quantizer-specific parameters."""
    # FSQ
    levels: list = field(default_factory=lambda: [8, 5, 5, 5])
    
    # VQ / SimVQ
    codebook_size: int = 8192
    commitment_weight: float = 1.0
    decay: float = 0.8

    # SimVQ specific
    use_mlp: bool = True
    
    # LFQ
    entropy_loss_weight: float = 0.1
    diversity_gamma: float = 1.0
    spherical: bool = True
    
    # LGQ (formerly LMB)
    num_bins: int = 128
    lambda_peak: float = 0.005
    lambda_bins: float = 0.005
    lambda_floor: float = 0.0
    p_min: float = 0.0
    tau: float = 1.0
    learn_widths: bool = False
    flatten_channels: bool = True
    init_method: str = "random"
    distance_type: str = "l2_sq"
    perchannel_fair: bool = False
    lmb_levels: list = field(default_factory=lambda: [16, 16, 8, 8])

    # Rotation trick (for rot_vq)
    rotation_trick: bool = False

    # SoftVQ
    softvq_tau: float = 0.07
    softvq_entropy_ratio: float = 0.01
    softvq_l2_norm: bool = True

    # IBQ
    ibq_beta: float = 0.25
    ibq_entropy_weight: float = 0.1
    ibq_entropy_temp: float = 0.01

    # BSQ
    bsq_beta: float = 0.25
    bsq_gamma0: float = 1.0
    bsq_gamma: float = 1.0
    bsq_zeta: float = 0.1
    bsq_group_size: int = 9
    bsq_l2_norm: bool = True
    bsq_inv_temperature: float = 1.0

    # iFSQ
    ifsq_alpha: float = 1.6


class UnifiedAutoEncoder(nn.Module):
    """
    Unified autoencoder supporting multiple quantization methods.
    
    All methods share the same Encoder/Decoder backbone,
    differing only in the quantization layer.
    
    Args:
        quantizer_type: One of "fsq", "vq", "lfq", "sim_vq", "lgq"
        model_config: Configuration for encoder/decoder architecture
        quantizer_config: Configuration for quantizer-specific parameters
    
    Example:
        model = UnifiedAutoEncoder("fsq", quantizer_config=QuantizerConfig(levels=[8,5,5,5]))
        model = UnifiedAutoEncoder("vq", quantizer_config=QuantizerConfig(codebook_size=8192))
        model = UnifiedAutoEncoder("lgq", quantizer_config=QuantizerConfig(num_bins=128))
    """
    
    def __init__(
        self,
        quantizer_type: QuantizerType,
        model_config: Optional[ModelConfig] = None,
        quantizer_config: Optional[QuantizerConfig] = None,
    ):
        super().__init__()
        
        self.quantizer_type = quantizer_type
        self.model_config = model_config or ModelConfig()
        self.quantizer_config = quantizer_config or QuantizerConfig()
        
        config = self.model_config
        qconfig = self.quantizer_config
        
        # Unified Encoder (shared by all quantizers)
        self.enc = Encoder(
            in_channels=config.in_channels,
            base_ch=config.base_ch,
            embedding_dim=config.embedding_dim,
            attn_res=config.attn_res,
        )
        
        # Quantizer-specific setup
        self._setup_quantizer(quantizer_type, config, qconfig)
        
        # Unified Decoder (shared by all quantizers)
        self.dec = Decoder(
            out_channels=config.in_channels,
            base_ch=config.base_ch,
            embedding_dim=config.embedding_dim,
            attn_res=config.attn_res,
            legacy_decoder=config.legacy_decoder,
        )
    
    def _setup_quantizer(self, qtype: str, config: ModelConfig, qconfig: QuantizerConfig):
        """Setup quantizer based on type."""
        if qtype == "fsq":
            from vector_quantize_pytorch import FSQ
            self.quantize_dim = len(qconfig.levels)
            self.num_codes = prod(qconfig.levels)
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = nn.Conv2d(config.embedding_dim, self.quantize_dim, 1)
            self.quantize = FSQ(qconfig.levels, channel_first=True)
            self.post_quant = nn.Conv2d(self.quantize_dim, config.embedding_dim, 1)

        elif qtype == "ifsq":
            from quantization.ifsq import iFSQ
            self.quantize_dim = len(qconfig.levels)
            self.num_codes = prod(qconfig.levels)
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = nn.Conv2d(config.embedding_dim, self.quantize_dim, 1)
            self.quantize = iFSQ(qconfig.levels, channel_first=True, alpha=qconfig.ifsq_alpha)
            self.post_quant = nn.Conv2d(self.quantize_dim, config.embedding_dim, 1)

        elif qtype == "vq":
            from vector_quantize_pytorch import VectorQuantize
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = None
            self.quantize = VectorQuantize(
                dim=config.embedding_dim,
                codebook_size=qconfig.codebook_size,
                channel_last=False,
                accept_image_fmap=True,
                commitment_weight=qconfig.commitment_weight,
                decay=qconfig.decay,
                rotation_trick=False,  # lib defaults rotation_trick=True when dim>1; plain VQ must use standard STE
                ema_update=False,
            )
            self.post_quant = None

        elif qtype == "rot_vq":
            from vector_quantize_pytorch import VectorQuantize
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = None
            self.quantize = VectorQuantize(
                dim=config.embedding_dim,
                codebook_size=qconfig.codebook_size,
                channel_last=False,
                accept_image_fmap=True,
                commitment_weight=qconfig.commitment_weight,
                decay=qconfig.decay,
                rotation_trick=True,
                ema_update=False,
            )
            self.post_quant = None

        elif qtype == "lfq":
            from vector_quantize_pytorch import LFQ
            assert log2(qconfig.codebook_size).is_integer(), "codebook_size must be power of 2"
            self.quantize_dim = int(log2(qconfig.codebook_size))
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = nn.Conv2d(config.embedding_dim, self.quantize_dim, 1)
            self.quantize = LFQ(
                dim=self.quantize_dim,
                channel_first=True,
                entropy_loss_weight=qconfig.entropy_loss_weight,
                diversity_gamma=qconfig.diversity_gamma,
                spherical=qconfig.spherical,
            )
            self.post_quant = nn.Conv2d(self.quantize_dim, config.embedding_dim, 1)
            
        elif qtype == "sim_vq":
            from vector_quantize_pytorch import SimVQ
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = None
            codebook_transform = None
            if qconfig.use_mlp:
                codebook_transform = nn.Sequential(
                    nn.Linear(config.embedding_dim, config.embedding_dim * 2),
                    nn.ReLU(),
                    nn.Linear(config.embedding_dim * 2, config.embedding_dim),
                )
            self.quantize = SimVQ(
                dim=config.embedding_dim,
                codebook_size=qconfig.codebook_size,
                channel_first=True,
                codebook_transform=codebook_transform,
            )
            self.post_quant = None
            self._commitment_weight = qconfig.commitment_weight
            
        elif qtype in ("lgq", "lmb"):
            from quantization.quantizers import MultiBinDiscretizer, MultiBinDiscretizerFair
            self.pre_quant = None
            self.post_quant = None
            if getattr(qconfig, "perchannel_fair", False) and getattr(qconfig, "lmb_levels", None):
                levels = list(qconfig.lmb_levels)
                self.quantize_dim = len(levels)
                self.num_codes = prod(levels)
                self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
                # Project 128 → 4 channels (16,16,8,8). Use a stable, deterministic
                # initialization to avoid collapsing the projection on first steps.
                self.pre_quant = nn.Conv2d(config.embedding_dim, self.quantize_dim, 1)
                with torch.no_grad():
                    self.pre_quant.weight.zero_()
                    self.pre_quant.bias.zero_()
                    copy_ch = min(self.quantize_dim, config.embedding_dim)
                    for i in range(copy_ch):
                        # Copy the first few encoder channels directly into the projected space.
                        self.pre_quant.weight[i, i, 0, 0] = 1.0
                # Check if we want flattened version (vector quantization over projected channels)
                if getattr(qconfig, "flatten_channels", False):
                    # Flattened per-channel fair: project to 4 channels, then vector quantize
                    self.quantize = MultiBinDiscretizer(
                        embedding_dim=self.quantize_dim,
                        num_bins=self.num_codes,
                        tau=qconfig.tau,
                        lambda_peak=qconfig.lambda_peak,
                        lambda_bins=qconfig.lambda_bins,
                        lambda_floor=qconfig.lambda_floor,
                        p_min=qconfig.p_min,
                        learn_widths=qconfig.learn_widths,
                        flatten_channels=True,
                        init_method=qconfig.init_method,
                        distance_type=qconfig.distance_type,
                    )
                else:
                    # Per-channel scalar quantization (original fair mode)
                    self.quantize = MultiBinDiscretizerFair(
                        levels=levels,
                        tau=qconfig.tau,
                        lambda_peak=qconfig.lambda_peak,
                        lambda_bins=qconfig.lambda_bins,
                        lambda_floor=qconfig.lambda_floor,
                        p_min=qconfig.p_min,
                        init_method=qconfig.init_method,
                    )
                self.post_quant = nn.Conv2d(self.quantize_dim, config.embedding_dim, 1)
            else:
                if qconfig.flatten_channels:
                    self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
                else:
                    self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
                self.quantize = MultiBinDiscretizer(
                    embedding_dim=config.embedding_dim,
                    num_bins=qconfig.num_bins,
                    tau=qconfig.tau,
                    lambda_peak=qconfig.lambda_peak,
                    lambda_bins=qconfig.lambda_bins,
                    lambda_floor=qconfig.lambda_floor,
                    p_min=qconfig.p_min,
                    learn_widths=qconfig.learn_widths,
                    flatten_channels=qconfig.flatten_channels,
                    init_method=qconfig.init_method,
                    distance_type=qconfig.distance_type,
                )
            self._tau = qconfig.tau

        elif qtype == "softvq":
            from quantization.softvq import SoftVectorQuantizer
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = None
            self.post_quant = None
            self.quantize = SoftVectorQuantizer(
                codebook_size=qconfig.codebook_size,
                embedding_dim=config.embedding_dim,
                tau=qconfig.softvq_tau,
                entropy_loss_ratio=qconfig.softvq_entropy_ratio,
                l2_norm=qconfig.softvq_l2_norm,
            )

        elif qtype == "ibq":
            from quantization.ibq import IndexPropagationQuantizer
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = None
            self.post_quant = None
            self.quantize = IndexPropagationQuantizer(
                codebook_size=qconfig.codebook_size,
                embedding_dim=config.embedding_dim,
                beta=qconfig.ibq_beta,
                entropy_loss_weight=qconfig.ibq_entropy_weight,
                entropy_temperature=qconfig.ibq_entropy_temp,
            )

        elif qtype == "bsq":
            from quantization.bsq import BinarySphericalQuantizer
            self.pre_vq_norm = nn.LayerNorm(config.embedding_dim)
            self.pre_quant = None
            self.post_quant = None
            self.quantize = BinarySphericalQuantizer(
                embedding_dim=config.embedding_dim,
                beta=qconfig.bsq_beta,
                gamma0=qconfig.bsq_gamma0,
                gamma=qconfig.bsq_gamma,
                zeta=qconfig.bsq_zeta,
                group_size=qconfig.bsq_group_size,
                l2_norm=qconfig.bsq_l2_norm,
                inv_temperature=qconfig.bsq_inv_temperature,
            )

        else:
            raise ValueError(f"Unknown quantizer type: {qtype}")
    
    @property
    def tau(self):
        """Temperature for LGQ (for annealing)."""
        return getattr(self, '_tau', 1.0)
    
    @tau.setter
    def tau(self, value):
        self._tau = value
        if self.quantizer_type in ("lgq", "lmb"):
            self.quantize.set_tau(value)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent indices."""
        z = self.enc(x)
        B, C, H, W = z.shape
        
        if self.quantizer_type in ("fsq", "ifsq"):
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_norm = self.pre_quant(z_norm)
            _, indices = self.quantize(z_norm)
            return indices

        elif self.quantizer_type in ("vq", "rot_vq"):
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            _, indices, _ = self.quantize(z_norm)
            return indices

        elif self.quantizer_type == "lfq":
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_norm = self.pre_quant(z_norm)
            _, indices, _ = self.quantize(z_norm)
            return indices
            
        elif self.quantizer_type == "sim_vq":
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            _, indices, _ = self.quantize(z_norm)
            return indices
            
        elif self.quantizer_type in ("lgq", "lmb"):
            if self.pre_quant is not None:
                B, C, H, W = z.shape
                z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
                z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                z_norm = self.pre_quant(z_norm)
                z_bct = z_norm.view(B, -1, H * W)
                *_, idx, _ = self.quantize(z_bct)
                return idx
            z_bct, H, W = self._bchw_to_bct(z)
            B, C, T = z_bct.shape
            if self.quantize.flatten_channels:
                z_flat = z_bct.permute(0, 2, 1).contiguous().view(B * T, C)
                z_norm = self.pre_vq_norm(z_flat)
                z_norm_bct = z_norm.view(B, T, C).permute(0, 2, 1).contiguous()
            else:
                z_flat = z_bct.permute(0, 2, 1).contiguous().view(B * T, C)
                z_norm = self.pre_vq_norm(z_flat).view(B, T, C).permute(0, 2, 1).contiguous()
                z_norm_bct = z_norm
            *_, idx, _ = self.quantize(z_norm_bct)
            return idx

        elif self.quantizer_type in ("softvq", "ibq", "bsq"):
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            _, _, indices = self.quantize(z_norm)
            return indices

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Returns vary by quantizer type:
            FSQ: (recon, indices)
            VQ: (recon, indices, commit_loss)
            LFQ: (recon, indices, entropy_loss)
            SimVQ: (recon, indices, commit_loss)
            LGQ: (recon, indices, vq_loss, perplexity)
        """
        z = self.enc(x)
        
        if self.quantizer_type in ("fsq", "ifsq"):
            # Apply LayerNorm before quantization
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_norm = self.pre_quant(z_norm)
            z_q, indices = self.quantize(z_norm)
            z_q = self.post_quant(z_q)
            recon = self.dec(z_q)
            return recon, indices

        elif self.quantizer_type in ("vq", "rot_vq"):
            # Apply LayerNorm before quantization
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_q, indices, commit_loss = self.quantize(z_norm)
            recon = self.dec(z_q)
            return recon, indices, commit_loss

        elif self.quantizer_type == "lfq":
            # Apply LayerNorm before quantization
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_norm = self.pre_quant(z_norm)
            z_q, indices, entropy_loss = self.quantize(z_norm)
            z_q = self.post_quant(z_q)
            recon = self.dec(z_q)
            return recon, indices, entropy_loss
            
        elif self.quantizer_type == "sim_vq":
            # Apply LayerNorm before quantization
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_q, indices, commit_loss = self.quantize(z_norm)
            recon = self.dec(z_q)
            return recon, indices, commit_loss * self._commitment_weight
            
        elif self.quantizer_type in ("lgq", "lmb"):
            if self.pre_quant is not None:
                B, C, H, W = z.shape
                z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
                z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
                z_norm = self.pre_quant(z_norm)
                z_bct = z_norm.view(B, -1, H * W)
                vq_loss, q_bct, perplexity, *_, idx, _ = self.quantize(z_bct)
                q = q_bct.view(B, -1, H, W)
                q = self.post_quant(q)
                recon = self.dec(q)
                return recon, idx, vq_loss, perplexity
            z_bct, H, W = self._bchw_to_bct(z)
            B, C, T = z_bct.shape
            if self.quantize.flatten_channels:
                z_flat = z_bct.permute(0, 2, 1).contiguous().view(B * T, C)
                z_norm = self.pre_vq_norm(z_flat)
                z_norm_bct = z_norm.view(B, T, C).permute(0, 2, 1).contiguous()
            else:
                z_flat = z_bct.permute(0, 2, 1).contiguous().view(B * T, C)
                z_norm = self.pre_vq_norm(z_flat).view(B, T, C).permute(0, 2, 1).contiguous()
                z_norm_bct = z_norm
            vq_loss, q_bct, perplexity, *_, idx, _ = self.quantize(z_norm_bct)
            q = self._bct_to_bchw(q_bct, H, W)
            recon = self.dec(q)
            return recon, idx, vq_loss, perplexity

        elif self.quantizer_type in ("softvq", "ibq", "bsq"):
            # All three: LayerNorm → quantize(BCHW) → decode
            B, C, H, W = z.shape
            z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
            z_norm = self.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            z_q, loss, indices = self.quantize(z_norm)
            recon = self.dec(z_q)
            return recon, indices, loss

    @staticmethod
    def _bchw_to_bct(z: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Convert [B, C, H, W] to [B, C, T] where T = H*W."""
        B, C, H, W = z.shape
        return z.view(B, C, H * W), H, W

    @staticmethod
    def _bct_to_bchw(z_bct: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Convert [B, C, T] to [B, C, H, W]."""
        B, C, T = z_bct.shape
        return z_bct.view(B, C, H, W)


def create_model(
    quantizer_type: QuantizerType,
    base_ch: int = 128,
    embedding_dim: int = 256,
    **quantizer_kwargs,
) -> UnifiedAutoEncoder:
    """
    Create a unified autoencoder model.
    
    Args:
        quantizer_type: One of "fsq", "vq", "lfq", "sim_vq", "lgq"
        base_ch: Base channels for encoder/decoder
        embedding_dim: Latent dimension
        **quantizer_kwargs: Quantizer-specific parameters
    
    Examples:
        model = create_model("fsq", levels=[8, 5, 5, 5])
        model = create_model("vq", codebook_size=8192)
        model = create_model("lgq", num_bins=128)
    """
    model_config = ModelConfig(base_ch=base_ch, embedding_dim=embedding_dim)
    quantizer_config = QuantizerConfig(**quantizer_kwargs)
    return UnifiedAutoEncoder(quantizer_type, model_config, quantizer_config)


def count_model_parameters(model: nn.Module) -> dict:
    """Count encoder, decoder, and quantizer parameters."""
    enc_params = sum(p.numel() for p in model.enc.parameters())
    dec_params = sum(p.numel() for p in model.dec.parameters())
    
    quant_params = 0
    if hasattr(model, 'quantize'):
        quant_params = sum(p.numel() for p in model.quantize.parameters())
    
    proj_params = 0
    if hasattr(model, 'pre_quant') and model.pre_quant is not None:
        proj_params += sum(p.numel() for p in model.pre_quant.parameters())
    if hasattr(model, 'post_quant') and model.post_quant is not None:
        proj_params += sum(p.numel() for p in model.post_quant.parameters())
    if hasattr(model, 'pre_vq_norm'):
        proj_params += sum(p.numel() for p in model.pre_vq_norm.parameters())
    
    total = sum(p.numel() for p in model.parameters())
    
    return {
        'encoder': enc_params,
        'decoder': dec_params,
        'quantizer': quant_params,
        'projections': proj_params,
        'total': total,
    }

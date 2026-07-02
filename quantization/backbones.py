"""
Unified encoder and decoder architecture for all quantization methods.

Uses MaskGIT-style architecture with attention for 16× compression.
All models share this same backbone for fair comparison.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
    def __init__(self, ch: int, ch_out: int | None = None):
        super().__init__()
        ch_out = ch if ch_out is None else ch_out
        num_groups = min(32, ch) if ch >= 32 else 1
        num_groups_out = min(32, ch_out) if ch_out >= 32 else 1
        self.in_norm = nn.GroupNorm(num_groups, ch)
        self.conv1 = nn.Conv2d(ch, ch_out, 3, 1, 1)
        self.out_norm = nn.GroupNorm(num_groups_out, ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, 1, 1)
        self.skip = nn.Identity() if ch == ch_out else nn.Conv2d(ch, ch_out, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.in_norm(x)))
        h = self.conv2(F.silu(self.out_norm(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """Self-attention block for 2D feature maps."""
    def __init__(self, ch: int, heads: int = 4):
        super().__init__()
        heads = min(heads, max(1, ch // 4))
        if ch < 4:
            heads = 1
        assert ch % heads == 0, f"channels ({ch}) must be divisible by heads ({heads})"
        self.heads = heads
        num_groups = min(32, ch) if ch >= 32 else 1
        self.norm = nn.GroupNorm(num_groups, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        B, C, H, W = h.shape
        Hn, d = self.heads, C // self.heads
        q = self.q(h).view(B, Hn, d, H * W).transpose(2, 3)
        k = self.k(h).view(B, Hn, d, H * W)
        v = self.v(h).view(B, Hn, d, H * W).transpose(2, 3)
        att = torch.softmax((q @ k) / (d ** 0.5), dim=-1)
        out = (att @ v).transpose(2, 3).contiguous().view(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    """2× spatial downsampling."""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, 4, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2× spatial upsampling."""
    def __init__(self, ch_in: int, ch_out: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(ch_in, ch_out, 4, 2, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


# -----------------------------------------------------------------------------
# Unified Encoder/Decoder (16× compression with attention)
# -----------------------------------------------------------------------------
class Encoder(nn.Module):
    """
    Unified encoder: 256→128→64→32→16 (16× downsample).
    
    Args:
        in_channels: Input image channels (default: 3)
        base_ch: Base channel count (default: 128)
        embedding_dim: Output embedding dimension (default: 256)
        attn_res: Tuple of resolutions where attention is applied (default: (16,))
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 128,
        embedding_dim: int = 256,
        attn_res: tuple = (16,),
    ):
        super().__init__()
        ch = base_ch
        
        # Stem
        self.stem = nn.Conv2d(in_channels, ch, 3, 1, 1)
        
        # Stage 1: 256 -> 128
        self.block_256 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        self.down1 = Downsample(ch, ch * 2)
        ch *= 2
        
        # Stage 2: 128 -> 64
        self.block_128 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        self.down2 = Downsample(ch, ch)
        
        # Stage 3: 64 -> 32
        self.block_64 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        self.down3 = Downsample(ch, ch)
        
        # Stage 4: 32 -> 16
        self.block_32 = nn.Sequential(
            ResBlock(ch), ResBlock(ch),
            AttnBlock(ch) if 32 in attn_res else nn.Identity()
        )
        self.down4 = Downsample(ch, ch)
        
        # Final stage at 16×16
        self.block_16 = nn.Sequential(
            ResBlock(ch), ResBlock(ch),
            AttnBlock(ch) if 16 in attn_res else nn.Identity()
        )
        
        # Project to embedding dimension
        self.to_emb = nn.Conv2d(ch, embedding_dim, 1)
        # Normalize the latent so the encoder is forced to use all channels.
        # Without this, the 256->embedding_dim projection collapses to rank ~4
        # (see eigenvalue spectrum), which causes cb=4096 to collapse to active=1.
        emb_groups = math.gcd(8, embedding_dim)
        self.emb_norm = nn.GroupNorm(emb_groups, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.block_256(h); h = self.down1(h)
        h = self.block_128(h); h = self.down2(h)
        h = self.block_64(h);  h = self.down3(h)
        h = self.block_32(h);  h = self.down4(h)
        h = self.block_16(h)
        return self.emb_norm(self.to_emb(h))


class Decoder(nn.Module):
    """
    Unified decoder: 16→32→64→128→256 (16× upsample).
    
    Args:
        out_channels: Output image channels (default: 3)
        base_ch: Base channel count (default: 128)
        embedding_dim: Input embedding dimension (default: 256)
        attn_res: Tuple of resolutions where attention is applied (default: (16,))
    """
    def __init__(
        self,
        out_channels: int = 3,
        base_ch: int = 128,
        embedding_dim: int = 256,
        attn_res: tuple = (16,),
        legacy_decoder: bool = False,
    ):
        super().__init__()
        if legacy_decoder:
            # Old behaviour: ch = max(32, embedding_dim).  Starves the decoder
            # when embedding_dim < base_ch * 2 (e.g. 64 vs 256).
            ch = max(32, embedding_dim)
        else:
            # Mirror the encoder's internal width (base_ch * 2) so the decoder
            # has symmetric capacity.
            ch = base_ch * 2
        
        # Project from embedding
        self.from_emb = nn.Conv2d(embedding_dim, ch, 1)
        
        # Stage at 16×16
        self.block_16 = nn.Sequential(
            ResBlock(ch), ResBlock(ch),
            AttnBlock(ch) if 16 in attn_res else nn.Identity()
        )
        
        # Stage 1: 16 -> 32
        self.up1 = Upsample(ch, ch)
        self.block_32 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        
        # Stage 2: 32 -> 64
        self.up2 = Upsample(ch, ch)
        self.block_64 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        
        # Stage 3: 64 -> 128
        self.up3 = Upsample(ch, ch)
        self.block_128 = nn.Sequential(ResBlock(ch), ResBlock(ch))
        
        # Stage 4: 128 -> 256
        self.up4 = Upsample(ch, base_ch)
        self.block_256 = nn.Sequential(ResBlock(base_ch), ResBlock(base_ch))
        
        # Output conv
        self.out = nn.Conv2d(base_ch, out_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_emb(z)
        h = self.block_16(h); h = self.up1(h)
        h = self.block_32(h); h = self.up2(h)
        h = self.block_64(h); h = self.up3(h)
        h = self.block_128(h); h = self.up4(h)
        h = self.block_256(h)
        return self.out(h)

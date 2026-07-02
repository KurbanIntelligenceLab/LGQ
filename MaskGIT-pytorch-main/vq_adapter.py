"""
Adapter to use trained UnifiedAutoEncoder models with MaskGIT transformer.

This wraps our quantization models to match the interface expected by MaskGIT:
- encode(x) -> (quant_z, emb_loss, (perplexity, min_encodings, indices))
- decode(quant) -> reconstructed image
- codebook.embedding(indices) -> lookup vectors from indices
"""

import sys
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, '..')
from quantization.model import UnifiedAutoEncoder, ModelConfig, QuantizerConfig


class CodebookWrapper(nn.Module):
    """Wrapper to provide embedding lookup interface for MaskGIT."""
    
    def __init__(self, model: UnifiedAutoEncoder, embedding_dim: int, spatial_size: int = 8):
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        self.spatial_size = spatial_size
        self.qtype = model.quantizer_type
        
    def embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings from indices.
        
        Args:
            indices: (B, num_tokens) token indices
            
        Returns:
            vectors: (B, num_tokens, embedding_dim) embedding vectors
        """
        B = indices.shape[0]
        device = indices.device
        
        if self.qtype in ("vq", "rot_vq"):
            # VQ has direct codebook access
            # indices: (B, H*W) -> need to look up from quantizer
            codebook = self.model.quantize._codebook.embed[0]  # (codebook_size, dim)
            vectors = codebook[indices]  # (B, num_tokens, dim)
            return vectors
            
        elif self.qtype == "sim_vq":
            codebook = self.model.quantize.codebook  # property: code_transform(frozen_codebook)
            vectors = codebook[indices]
            return vectors
            
        elif self.qtype == "fsq":
            # FSQ: indices encode level combinations, need to decode to vectors
            # The quantizer can convert indices back to quantized values
            indices_flat = indices.reshape(-1)  # (B*num_tokens,)
            z_q = self.model.quantize.indices_to_codes(indices_flat)  # (B*num_tokens, quantize_dim)
            z_q = z_q.view(B, -1, z_q.shape[-1])  # (B, num_tokens, quantize_dim)
            
            # Project back to embedding dim
            # Reshape to apply post_quant conv: (B, quantize_dim, H, W)
            H = W = self.spatial_size
            z_q_spatial = z_q.view(B, H, W, -1).permute(0, 3, 1, 2)  # (B, quantize_dim, H, W)
            z_embed = self.model.post_quant(z_q_spatial)  # (B, embedding_dim, H, W)
            vectors = z_embed.view(B, self.embedding_dim, -1).permute(0, 2, 1)  # (B, num_tokens, embedding_dim)
            return vectors
            
        elif self.qtype == "lfq":
            # LFQ: binary codes, use indices_to_codes
            indices_flat = indices.reshape(-1)
            z_q = self.model.quantize.indices_to_codes(indices_flat)  # (B*num_tokens, quantize_dim)
            z_q = z_q.view(B, -1, z_q.shape[-1])  # (B, num_tokens, quantize_dim)
            
            # Project back to embedding dim
            H = W = self.spatial_size
            z_q_spatial = z_q.view(B, H, W, -1).permute(0, 3, 1, 2)
            z_embed = self.model.post_quant(z_q_spatial)
            vectors = z_embed.view(B, self.embedding_dim, -1).permute(0, 2, 1)
            return vectors
            
        elif self.qtype == "lmb":
            # LMB: lookup codebook (centers). Flattened mode: centers [K, C], indices [B, T]
            quant = self.model.quantize
            if getattr(quant, "flatten_channels", False):
                # Flattened: single index per token, centers [K, C]
                indices_flat = indices.reshape(-1).long()
                z_q = quant.centers[indices_flat]  # (B*T, C)
                vectors = z_q.view(B, -1, self.embedding_dim)  # (B, num_tokens, C)
            else:
                # Per-channel: indices would be [B, T, C] - not used by MaskGIT; treat as flattened index
                indices_flat = indices.reshape(-1).long()
                z_q = quant.centers[indices_flat]
                vectors = z_q.view(B, -1, self.embedding_dim)
            return vectors

        elif self.qtype == "softvq":
            # SoftVQ: hard one-hot lookup of (optionally l2-normalized) embedding table
            quant = self.model.quantize
            emb = quant.embedding.weight  # [K, C]
            if getattr(quant, "l2_norm", False):
                emb = torch.nn.functional.normalize(emb, dim=-1)
            indices_flat = indices.reshape(-1).long()
            z_q = emb[indices_flat]  # (B*T, C)
            vectors = z_q.view(B, -1, self.embedding_dim)
            return vectors

        elif self.qtype == "ibq":
            # IBQ: hard argmax lookup of embedding table [K, C]
            emb = self.model.quantize.embedding.weight  # [K, C]
            indices_flat = indices.reshape(-1).long()
            z_q = emb[indices_flat]
            vectors = z_q.view(B, -1, self.embedding_dim)
            return vectors
        else:
            raise ValueError(f"Unknown quantizer type: {self.qtype}")


class VQModelAdapter(nn.Module):
    """
    Adapter that wraps UnifiedAutoEncoder to match MaskGIT's VQModel interface.
    
    MaskGIT expects:
    - encode(x) -> (quant_z, emb_loss, (perplexity, min_encodings, indices))
    - decode(quant) -> image
    - codebook.embedding(indices) -> vectors
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        quantizer_type: str,
        device: str = "cuda",
        image_size: int = 128,
        **quantizer_kwargs
    ):
        super().__init__()

        self.quantizer_type = quantizer_type
        self.device = device
        
        # Load checkpoint to get actual config
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # Infer config from checkpoint weights
        state_dict = ckpt.get("model_state_dict", ckpt)
        
        # Get base_ch from stem weight shape
        stem_w = state_dict.get('enc.stem.weight')
        base_ch = stem_w.shape[0] if stem_w is not None else 64
        
        # Get embedding_dim from to_emb weight shape
        emb_w = state_dict.get('enc.to_emb.weight')
        embedding_dim = emb_w.shape[0] if emb_w is not None else 128
        
        # Create model config with correct dimensions
        model_config = ModelConfig(
            base_ch=base_ch,
            embedding_dim=embedding_dim,
        )
        
        # Try to get quantizer config from saved config
        if "config" in ckpt:
            cfg = ckpt["config"]
            _shared_keys = ['levels', 'codebook_size', 'num_bins', 'perchannel_fair',
                            'lmb_levels', 'flatten_channels',
                            'softvq_tau', 'softvq_entropy_ratio', 'softvq_l2_norm',
                            'ibq_beta', 'ibq_entropy_weight', 'ibq_entropy_temp']
            if isinstance(cfg, dict):
                for k in _shared_keys:
                    if k in cfg and k not in quantizer_kwargs:
                        quantizer_kwargs[k] = cfg[k]
            else:
                for k in _shared_keys:
                    if hasattr(cfg, k) and k not in quantizer_kwargs:
                        quantizer_kwargs[k] = getattr(cfg, k)
        
        quantizer_config = QuantizerConfig(**quantizer_kwargs)
        
        # Create model
        self.model = UnifiedAutoEncoder(quantizer_type, model_config, quantizer_config)
        
        # Load weights (state_dict already extracted above)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # CPU probe — first forward on cuda right after construction
        # segfaults inside cuDNN on H100 + torch 2.2.0+cu118.
        with torch.no_grad():
            probe = torch.zeros(1, 3, image_size, image_size)
            z_probe = self.model.enc(probe)
        self.spatial_size = z_probe.shape[-1]
        self.num_image_tokens = self.spatial_size ** 2
        self.embedding_dim = embedding_dim

        self.model = self.model.to(device)
        
        # Create codebook wrapper
        self.codebook = CodebookWrapper(self.model, embedding_dim, self.spatial_size)
        
        print(f"Loaded {quantizer_type} model from {checkpoint_path}")
        print(f"  Embedding dim: {embedding_dim}")
        print(f"  Spatial size: {self.spatial_size}x{self.spatial_size} = {self.num_image_tokens} tokens")
        
    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """
        Encode images to quantized latents and indices.
        
        Args:
            x: (B, 3, H, W) input images in [-1, 1]
            
        Returns:
            quant_z: (B, embedding_dim, h, w) quantized latents
            emb_loss: scalar embedding loss (0 for frozen model)
            info: tuple of (perplexity, min_encodings, indices)
                  where indices is (B*h*w,) or (B, h*w)
        """
        B = x.shape[0]
        
        # Normalize from [-1, 1] to [0, 1] if needed (our models expect [0, 1])
        if x.min() < 0:
            x = (x + 1) / 2
        
        # Get encoder output
        z = self.model.enc(x)  # (B, C, H, W)
        _, C, H, W = z.shape
        
        # Apply pre-vq normalization
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(B * H * W, C)
        z_norm = self.model.pre_vq_norm(z_flat).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
        # Apply pre_quant projection if present
        if self.model.pre_quant is not None:
            z_norm = self.model.pre_quant(z_norm)
        
        # Get quantized output and indices based on quantizer type
        if self.quantizer_type == "fsq":
            z_q, indices = self.model.quantize(z_norm)
            if self.model.post_quant is not None:
                z_q = self.model.post_quant(z_q)
            emb_loss = torch.tensor(0.0, device=x.device)
            perplexity = torch.tensor(0.0, device=x.device)
            
        elif self.quantizer_type in ("vq", "rot_vq"):
            z_q, indices, commit_loss = self.model.quantize(z_norm)
            emb_loss = commit_loss
            perplexity = torch.tensor(0.0, device=x.device)
            
        elif self.quantizer_type == "lfq":
            z_q, indices, entropy_loss = self.model.quantize(z_norm)
            if self.model.post_quant is not None:
                z_q = self.model.post_quant(z_q)
            emb_loss = entropy_loss
            perplexity = torch.tensor(0.0, device=x.device)
            
        elif self.quantizer_type == "sim_vq":
            z_q, indices, commit_loss = self.model.quantize(z_norm)
            emb_loss = commit_loss
            perplexity = torch.tensor(0.0, device=x.device)
            
        elif self.quantizer_type == "lmb":
            # LMB needs special handling
            if self.model.pre_quant is not None:
                z_bct = z_norm.view(B, -1, H * W)
                vq_loss, q_bct, perplexity, *_, indices, _ = self.model.quantize(z_bct)
                z_q = q_bct.view(B, -1, H, W)
                if self.model.post_quant is not None:
                    z_q = self.model.post_quant(z_q)
            else:
                z_bct = z_norm.view(B, C, H * W)
                vq_loss, q_bct, perplexity, *_, indices, _ = self.model.quantize(z_bct)
                z_q = q_bct.view(B, C, H, W)
            emb_loss = vq_loss

        elif self.quantizer_type == "softvq":
            # SoftVQ: pre_quant=None, post_quant=None, quantize takes BCHW
            z_q, entropy_loss, indices = self.model.quantize(z_norm)
            emb_loss = entropy_loss
            perplexity = torch.tensor(0.0, device=x.device)

        elif self.quantizer_type == "ibq":
            # IBQ: pre_quant=None, post_quant=None, quantize takes BCHW -> (z_q, loss, indices[B,T])
            z_q, total_loss, indices = self.model.quantize(z_norm)
            emb_loss = total_loss
            perplexity = torch.tensor(0.0, device=x.device)
        else:
            raise ValueError(f"Unknown quantizer type: {self.quantizer_type}")
        
        # Ensure indices are in expected format (B, num_tokens) or (B*num_tokens,)
        indices = indices.view(B, -1)  # (B, H*W)
        
        # Return in MaskGIT expected format
        min_encodings = None  # Not needed for generation
        return z_q, emb_loss, (perplexity, min_encodings, indices)
    
    @torch.no_grad()  
    def decode(self, quant: torch.Tensor) -> torch.Tensor:
        """
        Decode quantized latents to images.
        
        Args:
            quant: (B, embedding_dim, H, W) quantized latents
            
        Returns:
            images: (B, 3, H*16, W*16) reconstructed images in [-1, 1]
        """
        images = self.model.dec(quant)
        # Convert back to [-1, 1] range for MaskGIT
        images = images * 2 - 1
        return images
    
    def forward(self, x: torch.Tensor):
        """Full forward pass for compatibility."""
        quant, emb_loss, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, emb_loss


def load_vq_adapter(
    quantizer_type: str,
    checkpoint_path: str,
    device: str = "cuda",
    image_size: int = 128,
) -> VQModelAdapter:
    """
    Load a VQ adapter for use with MaskGIT.
    
    Args:
        quantizer_type: One of "vq", "fsq", "lfq", "sim_vq", "lmb"
        checkpoint_path: Path to saved checkpoint
        device: Device to load model on
        
    Returns:
        VQModelAdapter instance
    """
    # Load checkpoint to extract config
    import torch as _torch
    ckpt = _torch.load(checkpoint_path, map_location="cpu")
    
    # Default quantizer kwargs for 16k codebook
    quantizer_kwargs = {}
    
    # Try to get config from checkpoint
    if "config" in ckpt:
        cfg = ckpt["config"]
        if isinstance(cfg, dict):
            # Extract relevant quantizer config
            for k in ['levels', 'codebook_size', 'num_bins', 'perchannel_fair',
                      'lmb_levels', 'flatten_channels', 'use_mlp', 'lambda_peak',
                      'lambda_bins', 'lambda_floor', 'p_min', 'learn_widths', 'init_method',
                      'softvq_tau', 'softvq_entropy_ratio', 'softvq_l2_norm',
                      'ibq_beta', 'ibq_entropy_weight', 'ibq_entropy_temp']:
                if k in cfg:
                    quantizer_kwargs[k] = cfg[k]
            print(f"Loaded config from checkpoint: {quantizer_kwargs}")
    
    # Fallback defaults if not in checkpoint
    if not quantizer_kwargs:
        if quantizer_type == "vq":
            quantizer_kwargs = {"codebook_size": 16384}
        elif quantizer_type == "fsq":
            quantizer_kwargs = {"levels": [16, 16, 8, 8]}  # 16384 codes
        elif quantizer_type == "lfq":
            quantizer_kwargs = {"codebook_size": 16384}  # 2^14
        elif quantizer_type == "sim_vq":
            quantizer_kwargs = {"codebook_size": 16384, "use_mlp": True}
        elif quantizer_type in ("lmb", "lmb_fixed"):
            quantizer_kwargs = {
                "num_bins": 16384,
                "flatten_channels": True,
            }
        elif quantizer_type == "softvq":
            quantizer_kwargs = {"codebook_size": 16384}
        elif quantizer_type == "ibq":
            quantizer_kwargs = {"codebook_size": 16384}
    
    # lmb_fixed uses same model as lmb
    model_type = "lmb" if quantizer_type == "lmb_fixed" else quantizer_type
    return VQModelAdapter(
        checkpoint_path=checkpoint_path,
        quantizer_type=model_type,
        device=device,
        image_size=image_size,
        **quantizer_kwargs
    )


# Quick test
if __name__ == "__main__":
    import os
    
    # Test with VQ checkpoint
    ckpt_path = "../results/vq/vq_cb16k/checkpoints/best_model.pt"
    if os.path.exists(ckpt_path):
        print("Testing VQ adapter...")
        adapter = load_vq_adapter("vq", ckpt_path, device="cpu")
        
        # Test encode/decode
        x = torch.randn(2, 3, 128, 128)
        quant, loss, (perp, _, indices) = adapter.encode(x)
        print(f"Encoded shape: {quant.shape}")
        print(f"Indices shape: {indices.shape}")
        
        # Test decode
        dec = adapter.decode(quant)
        print(f"Decoded shape: {dec.shape}")
        
        # Test codebook lookup
        vectors = adapter.codebook.embedding(indices)
        print(f"Vectors shape: {vectors.shape}")

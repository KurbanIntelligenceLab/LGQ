#!/usr/bin/env python3
"""
Unified usage example for all quantization methods.

Shows how to create and use UnifiedAutoEncoder with different quantizers.
"""

import torch
from quantization import UnifiedAutoEncoder, ModelConfig, QuantizerConfig, create_model


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create dummy input (batch of 256x256 RGB images)
    x = torch.randn(2, 3, 256, 256).to(device)
    
    print("=" * 60)
    print("Unified AutoEncoder Example")
    print("=" * 60)
    print()
    
    # Common model config (same for all quantizers)
    model_config = ModelConfig(base_ch=128, embedding_dim=256)
    
    # Example 1: FSQ (Finite Scalar Quantization)
    print("1. FSQ Model")
    print("-" * 40)
    fsq_model = UnifiedAutoEncoder(
        "fsq",
        model_config,
        QuantizerConfig(levels=[8, 5, 5, 5]),
    ).to(device)
    
    recon, indices = fsq_model(x)
    print(f"   Input shape: {x.shape}")
    print(f"   Recon shape: {recon.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Active codes: {indices.unique().numel()}")
    print()
    
    # Example 2: VQ (Vector Quantization)
    print("2. VQ Model")
    print("-" * 40)
    vq_model = UnifiedAutoEncoder(
        "vq",
        model_config,
        QuantizerConfig(codebook_size=8192),
    ).to(device)
    
    recon, indices, commit_loss = vq_model(x)
    print(f"   Recon shape: {recon.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Commit loss: {commit_loss.item():.4f}")
    print()
    
    # Example 3: LFQ (Lookup-Free Quantization)
    print("3. LFQ Model")
    print("-" * 40)
    lfq_model = UnifiedAutoEncoder(
        "lfq",
        model_config,
        QuantizerConfig(codebook_size=65536),
    ).to(device)
    
    recon, indices, entropy_loss = lfq_model(x)
    print(f"   Recon shape: {recon.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Entropy loss: {entropy_loss.item():.4f}")
    print()
    
    # Example 4: SimVQ
    print("4. SimVQ Model")
    print("-" * 40)
    sim_vq_model = UnifiedAutoEncoder(
        "sim_vq",
        model_config,
        QuantizerConfig(codebook_size=8192, use_mlp=True),
    ).to(device)
    
    recon, indices, commit_loss = sim_vq_model(x)
    print(f"   Recon shape: {recon.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   Commit loss: {commit_loss.item():.4f}")
    print()
    
    # Example 5: LMB (Learnable Multi-Bin)
    print("5. LMB Model")
    print("-" * 40)
    lmb_model = UnifiedAutoEncoder(
        "lmb",
        model_config,
        QuantizerConfig(num_bins=128),
    ).to(device)
    
    recon, indices, vq_loss, perplexity = lmb_model(x)
    print(f"   Recon shape: {recon.shape}")
    print(f"   Indices shape: {indices.shape}")
    print(f"   VQ loss: {vq_loss.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.1f}")
    print()
    
    # Example 6: Using create_model factory function
    print("6. Factory Function (create_model)")
    print("-" * 40)
    model = create_model("fsq", levels=[8, 5, 5, 5]).to(device)
    recon, _ = model(x)
    print(f"   Created FSQ model with factory function")
    print(f"   Recon shape: {recon.shape}")
    print()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

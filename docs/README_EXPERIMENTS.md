# ImageNet Vector Quantization Experiments

Complete pipeline for training and evaluating 5 vector quantization methods on ImageNet with **fair, standardized comparison**.

## Methods Included

1. **VQ** - Traditional Vector Quantization (VQ-VAE)
2. **FSQ** - Finite Scalar Quantization
3. **LFQ** - Lookup Free Quantization
4. **SimVQ** - Simple Vector Quantization
5. **LMB-VAE** - Learnable Multi-Bin Discretization (NEW)

## Fair Comparison Guarantee

All methods use **identical**:
- ✅ Architecture: Simple 3-layer CNN
- ✅ Compression: 8× (28×28 latent from 224×224 input)
- ✅ Feature dimension: 256 channels
- ✅ Reconstruction loss: L1 (MAE)
- ✅ Optimizer: AdamW with lr=3e-4
- ✅ Batch size: 64
- ✅ Training epochs: 100
- ✅ Data normalization: ImageNet standard

## Quick Start

### 1. Train All Methods (Parallel)
```bash
bash RUN_ALL_EXPERIMENTS.sh
```

This runs all 5 methods in parallel on different GPUs with fair configuration.

### 2. Evaluate All Methods
```bash
bash RUN_ALL_EVALUATIONS.sh
```

Computes all metrics: PSNR, SSIM, LPIPS, FID, codebook usage.

### 3. Compare Results
```bash
# View all metrics
for method in vq fsq lfq sim_vq lmb; do
    echo "=== $method ==="
    cat results/${method}/metrics.json
done
```

## Individual Training

### Standard Commands
```bash
# All use same parameters except method-specific quantizer settings
./scripts/shell/train_vq.sh --gpu 0      # VQ: codebook_size=8192
./scripts/shell/train_fsq.sh --gpu 1     # FSQ: levels=[8,5,5,5]
./scripts/shell/train_lfq.sh --gpu 2     # LFQ: codebook_size=65536
./scripts/shell/train_sim_vq.sh --gpu 3  # SimVQ: codebook_size=8192
./scripts/shell/train_lmb.sh --gpu 4     # LMB: embedding_dim=256, num_bins=25
```

## Project Structure

```
vector-quantize-pytorch/
├── Quantization/              # LMB-VAE module (modular)
│   ├── losses.py              # 15 lines
│   ├── quantizers.py          # 177 lines
│   ├── backbones.py           # 260 lines
│   ├── model.py               # 122 lines
│   └── README.md
├── scripts/
│   ├── python/                # 5 training + 1 evaluation scripts
│   └── shell/                 # 10 wrapper scripts
├── RUN_ALL_EXPERIMENTS.sh     # Train all methods
├── RUN_ALL_EVALUATIONS.sh     # Evaluate all methods
├── FAIR_COMPARISON.md         # Fairness documentation
├── COMPARISON_TABLE.md        # Quick reference
├── IMAGENET_EVALUATION.md     # Complete guide
└── README_EXPERIMENTS.md      # This file
```

## Evaluation Metrics

| Metric | What It Measures | Better |
|--------|-----------------|--------|
| **PSNR** | Signal-to-noise ratio | Higher |
| **SSIM** | Structural similarity | Higher |
| **LPIPS** | Perceptual similarity | Lower |
| **FID** | Distribution similarity | Lower |
| **Codebook Usage** | % codes used | Higher |
| **Reconstruction Loss** | Pixel-level error | Lower |

## Expected Outputs

```
results/
├── vq/
│   ├── checkpoints/best_model.pt
│   ├── config.json
│   ├── train_history.json
│   ├── metrics.json
│   └── samples/
├── fsq/
├── lfq/
├── sim_vq/
└── lmb/
```

## Method Comparison

### Quantization Approaches

| Method | Type | Learnable | Codebook Size | Parameters |
|--------|------|-----------|---------------|------------|
| VQ | Vector | EMA | 8,192 | 2.1M |
| FSQ | Scalar | No | 1,000 | 0 |
| LFQ | Binary | No | 65,536 | 0 |
| SimVQ | Implicit | Yes | 8,192 | Implicit |
| LMB | Per-channel | Yes | 25^256 | 6,400 |

### Key Innovations

- **VQ**: Nearest neighbor in shared vector space
- **FSQ**: Fixed scalar levels, no learning
- **LFQ**: Binary codes with entropy regularization
- **SimVQ**: Implicit codes via learned projection
- **LMB**: Per-channel learnable bin centers

## Documentation

- [`FAIR_COMPARISON.md`](FAIR_COMPARISON.md) - Fairness details
- [`COMPARISON_TABLE.md`](COMPARISON_TABLE.md) - Quick reference
- [`IMAGENET_EVALUATION.md`](IMAGENET_EVALUATION.md) - Complete guide
- [`Quantization/README.md`](Quantization/README.md) - LMB-VAE module docs

## Installation

```bash
# Core + evaluation dependencies
pip install -e ".[evaluation]"

# Or separately
pip install torch torchvision tqdm
pip install lpips pytorch-fid scikit-image pillow
```

## Hardware Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (for batch size 64)
- **Storage**: ~150GB for ImageNet + ~50GB per model checkpoints
- **RAM**: 32GB+ recommended for data loading

## Troubleshooting

**Out of memory**: Reduce batch size to 32 or 16
**Slow training**: Check data loading (increase num_workers)
**FID fails**: Ensure pytorch-fid installed and sufficient disk space

Ready for comprehensive ImageNet experiments! 🚀


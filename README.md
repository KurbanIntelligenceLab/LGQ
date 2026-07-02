# LGQ: Learnable Geometric Quantization for Image Tokenization

Official code for **"LGQ: Learnable Geometric Quantization for Image Tokenization — An
Equal-Budget Comparison of Vector Quantizers on ImageNet."**

Recent quantizers (FSQ, LFQ, BSQ) achieve collapse-free training by abandoning the
learnable codebook in favor of engineered geometries (fixed scalar grids or binary
lattices). **LGQ (Learnable Geometric Quantization)** keeps a *learnable* codebook of
per-bin centers with explicit usage regularization, so it can adapt to the data while
remaining collapse-resistant as the codebook grows. This repository provides a single,
fair training/evaluation harness that benchmarks LGQ against five baselines under an
**equal codebook budget** on ImageNet, and sweeps the codebook size
`K ∈ {4096, 8192, 16384, 32768, 65536}`.

> **Note on naming.** LGQ was developed internally under the name *LMB* ("Learnable
> Multi-Bin"). The code selector is now `--model lgq`; `--model lmb` is kept as a
> backward-compatible alias, and a number of internal ablation/analysis scripts still use
> the `lmb` tag.

This project is built on top of
[`lucidrains/vector-quantize-pytorch`](https://github.com/lucidrains/vector-quantize-pytorch)
(MIT), which is vendored under `external/vector_quantize_pytorch/` and supplies the FSQ,
VQ, LFQ, and SimVQ baseline implementations. See [Acknowledgements](#acknowledgements).

---

## Contents

- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Training](#training)
- [Evaluation](#evaluation)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Installation

Python ≥ 3.9 (developed and tested on **3.11**) and PyTorch ≥ 2.4.

```bash
git clone https://github.com/KurbanIntelligenceLab/vector-quantize.git
cd vector-quantize

python -m venv .venv && source .venv/bin/activate
pip install -e ".[training,evaluation]"
```

The `training` extra pulls in `torchvision`, `tqdm`, `matplotlib`, and `numpy`; the
`evaluation` extra adds `lpips`, `pytorch-fid`, `scikit-image`, `pillow`, and `scipy` for
rFID/LPIPS/SSIM. `wandb` is optional and used only when `--wandb` is passed.

All modules assume the repo root and `external/` are on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD:$PWD/external"
```

## Data preparation

Training uses ImageNet by default. Organize it in the standard `torchvision`
`ImageFolder` layout:

```
$DATA_ROOT/
  train/<synset>/*.JPEG
  val/<synset>/*.JPEG
```

Pass the location with `--data-root $DATA_ROOT`. Supported `--dataset` values are
`imagenet` (default), `imagenet100` (first 100 synsets), and `cifar10` (auto-downloaded).
Images are resized (bicubic) and center-cropped to `--image-size` and normalized with the
standard ImageNet statistics.

## Training

All methods share the **same encoder/decoder backbone** and differ only in the
quantization layer, so comparisons are equal-budget by construction. The single entry
point is `scripts/train.py`; select the method with `--model`:

| `--model` | Method |
|-----------|--------|
| `lgq`     | **Learnable Geometric Quantization (ours)** |
| `vq`      | Vector Quantization (VQ-VAE) |
| `rot_vq`  | Rotation-trick VQ |
| `fsq`     | Finite Scalar Quantization |
| `lfq`     | Lookup-Free Quantization |
| `bsq`     | Binary Spherical Quantization |
| `sim_vq`, `ibq`, `softvq` | additional baselines |

**Train LGQ** (single GPU, codebook size `K = 16384`):

```bash
python scripts/train.py \
  --model lgq \
  --data-root "$DATA_ROOT" --dataset imagenet --image-size 256 \
  --num-bins 16384 \
  --dim 256 --embedding-dim 64 \
  --lambda-peak 0.005 --lambda-bins 0.005 \
  --tau-start 1.0 --tau-end 0.1 --flatten-channels \
  --batch-size 64 --epochs 100 --lr 3e-4 \
  --perceptual-weight 1.0 --gan-weight 0.1
```

**Train a baseline** at the matched budget, e.g. VQ / FSQ / LFQ:

```bash
python scripts/train.py --model vq  --codebook-size 16384 --data-root "$DATA_ROOT"
python scripts/train.py --model fsq --data-root "$DATA_ROOT"
python scripts/train.py --model lfq --codebook-size 16384 --data-root "$DATA_ROOT"
```

Run `python scripts/train.py --help` for the full argument list, including LGQ-specific
options (`--num-bins`, `--lambda-peak`, `--lambda-bins`, `--tau-start/-end`,
`--tau-schedule`, `--init-method`, `--distance-type`, `--perchannel-fair`,
`--lgq-levels`). Outputs (checkpoints, `config.json`, `train_metrics.csv`,
`eval_metrics.csv`, `train_history.json`) are written under `results/<run-name>/`.

### Multi-GPU / SLURM

Ready-to-submit SLURM launchers live in `shell/slurm/`. The canonical LGQ launcher is
`shell/slurm/train_lgq.sh` (4×GPU DDP; codebook size via the `CODEBOOK_SIZE` env var),
with matching `train_vq.sh`, `train_fsq.sh`, `train_lfq.sh`, `train_bsq.sh`, etc. for the
baselines. Edit the `#SBATCH` account/partition lines for your cluster, then:

```bash
CODEBOOK_SIZE=16384 sbatch shell/slurm/train_lgq.sh
```

## Evaluation

Reconstruction metrics (rFID, PSNR, SSIM, LPIPS, codebook utilization, perplexity) are
computed after every epoch during training and can be recomputed from a checkpoint:

```bash
python scripts/evaluation/evaluate.py \
  --checkpoint results/<run-name>/checkpoints/best_model.pt \
  --data-root "$DATA_ROOT" --split val
```

Class-conditional **generation** metrics (gFID, Inception Score) are produced by training
a MaskGIT transformer on top of a frozen tokenizer; see `MaskGIT-pytorch-main/`.

## Reproducing the paper

1. Train LGQ and the baselines at each codebook size in
   `K ∈ {4096, 8192, 16384, 32768, 65536}` (see `shell/slurm/`).
2. Aggregate the per-run metric CSVs into comparison tables with the scripts under
   `scripts/evaluation/` and `scripts/tables/`.
3. Regenerate the paper's figures with the scripts under `scripts/plotting/` and the
   codebook/geometry analyses under `scripts/analysis/`.

## Repository layout

```
quantization/        Unified autoencoder + quantizers (LGQ = MultiBinDiscretizer)
configs/             Per-method configs; select via --model (lgq, vq, fsq, lfq, ...)
losses/              Perceptual + GAN (discriminator) losses
scripts/
  train.py           Training entry point
  evaluation/        Checkpoint evaluation, rFID/metrics
  analysis/          Codebook usage, geometry, and utilization analyses
  plotting/          Paper figures
  tables/            LaTeX/CSV table generation
shell/               Launchers (shell/slurm/ for cluster submission)
external/            Vendored lucidrains/vector-quantize-pytorch (baselines)
MaskGIT-pytorch-main/  Class-conditional generation (gFID / IS)
tests/               Unit tests (pytest)
```

## Citation

```bibtex
@inproceedings{lgq2026,
  title     = {LGQ: Learnable Geometric Quantization for Image Tokenization},
  author    = {Anonymous},
  booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  year      = {2026}
}
```

Please also cite the underlying VQ library (below) if you use the baseline quantizers.

## Acknowledgements

The baseline quantizers (FSQ, VQ, LFQ, SimVQ) are provided by
[`lucidrains/vector-quantize-pytorch`](https://github.com/lucidrains/vector-quantize-pytorch),
vendored under `external/vector_quantize_pytorch/` and used under its MIT license (see
[`LICENSE`](LICENSE)).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

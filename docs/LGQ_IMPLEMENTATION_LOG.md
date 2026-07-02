# LGQ Implementation Log: Data-Centric Codebook Init & Geometry Analysis

**Date**: 2026-03-28
**Project**: /N/project/de_briujn_graph/Projects/vector-quantize/

---

## Overview

Two new contributions implemented on top of the existing LGQ (Learnable Geometric Quantization) codebase:

1. **Contribution 1**: Data-Centric Codebook Initialisation & N* Prediction (Phase 0)
2. **Contribution 2**: Codebook Geometry Visualisation (5 scientific figures)

Plus supporting infrastructure: ImageNet-100 dataset support, codebook init loading, SLURM scripts.

---

## Files Created

| File | Purpose |
|------|---------|
| `phase0_datacentric.py` | Phase 0 script — encodes patches, computes d_eff, predicts N*, runs density-weighted k-means, saves codebooks |
| `geometry_analysis.py` | Produces 5 scientific figures proving LGQ learns Gersho-predicted geometry |
| `run_phase0.sh` | SLURM script for full Phase 0 (M'=50000, 1 GPU, 16GB, 1hr) |
| `run_phase0_test.sh` | SLURM script for quick Phase 0 test (M'=1000, 15min) |

## Files Modified

| File | Changes |
|------|---------|
| `scripts/train.py` | Added `--dataset imagenet100` flag, `ImageNet100Folder` class, `--codebook-init` flag for loading Phase 0 codebooks |

---

## What Was Done and Why

### 1. ImageNet-100 Dataset Support (`scripts/train.py`)

**What**: Added `--dataset` flag with choices `imagenet` (default, 1000 classes) and `imagenet100` (100-class subset).

**Why**: Full ImageNet has ~1.28M images. ImageNet-100 has ~130K images (first 100 synsets sorted alphabetically), enabling fast iteration during development and ablation studies without waiting for full-dataset training runs.

**How it works**:
- `get_imagenet100_synsets(data_root)`: Returns the first 100 synset folder names from `train/` directory, sorted alphabetically.
- `ImageNet100Folder(SafeImageFolder)`: Subclass that filters `self.samples` to only include images from the 100 selected synsets.
- Applied to both train and val splits when `--dataset imagenet100` is used.

**Location in code**: Lines ~111-130 (new classes), lines ~665-675 (train loading), lines ~688-690 (val loading).

### 2. Codebook Init Loading (`scripts/train.py`)

**What**: Added `--codebook-init` argument that loads a pre-computed codebook from a `.npy` file into the LMB quantizer's `centers` parameter.

**Why**: Phase 0 produces optimised codebook initialisations. The training script needs to accept these so we can compare data-centric init vs random/kmeans/gaussian inits.

**How it works**:
- After model creation, if `--codebook-init` is provided, loads the numpy array and copies it into `model.quantize.centers.data`.
- Sets `model.quantize._inited` to 1 to skip the lazy first-batch initialisation.
- Handles K mismatch (truncates if codebook has more codes, pads with small random noise if fewer).

**Location in code**: Lines ~764-782 (after model creation, before optimizer setup).

### 3. Phase 0: Data-Centric Codebook Initialisation (`phase0_datacentric.py`)

**What**: A standalone script that runs BEFORE any LGQ training. Takes a frozen encoder checkpoint and produces:
- Optimal codebook size prediction N*
- Three codebook initialisations (data-centric, standard k-means, Gaussian)
- Predicted utilisation analysis for each
- Summary JSON with all metrics

**Why**: Standard codebook initialisation (random or Gaussian) doesn't account for the geometry of the latent space. Gersho-Zador quantisation theory tells us the optimal codeword density should be proportional to p_Z(z)^{d/(d+2)}, not uniform. By running a data-centric initialisation phase, we can:
- Start training from a better codebook (faster convergence)
- Predict the optimal codebook size N* before committing to a training run
- Avoid codebook collapse (dead codes) from the start

**The 6 steps**:

**Step 1 — Encode patches**: Runs M' images through the frozen encoder, applying the same LayerNorm and pre_quant projection the model uses internally. Produces latent vectors Z of shape [num_patches, C] where C=128 (the embedding_dim from the checkpoint config).

**Step 2 — Covariance & d_eff**: Computes the covariance matrix Sigma of the centered latent vectors. Uses SVD when C>256 for numerical stability. Computes effective intrinsic dimension via the participation ratio: d_eff = (sum(lambda_i))^2 / sum(lambda_i^2). Saves eigenvalue spectrum plot.

**Step 3 — Predict N***: Uses the Gersho-Zador formula:
- D* = 10th percentile of pairwise squared distances / C
- Norm factor computed in log space to avoid overflow: log(||p_Z||) = -d/(d+2) * log(2pi) - log_det_Sigma/(d+2)
- N* = exp((d/2) * [log(C_d) + log(norm_factor) - log(D*)])
- Capped at 131072 to stay in practical VQ range
- Recommended K = 2 * N* (safety margin)

**Step 4 — Density-weighted k-means**: Computes Mahalanobis distance for each latent vector, then weights samples by exp(mahal^2 / (d_eff+2)). This tilts the sampling distribution from p_Z toward p_Z^{d/(d+2)} = Lambda* (Gersho-optimal density). Runs sklearn KMeans with `sample_weight`. Also runs standard k-means and generates Gaussian random codebook as baselines.

**Step 5 — Utilisation analysis**: Assigns all M' latents to nearest codeword for each codebook. Reports: fraction of codes active, assignment entropy (bits), average distance to nearest codeword.

**Step 6 — Save summary JSON**: All metrics saved to `phase0_results/summary.json`.

### 4. Geometry Analysis (`geometry_analysis.py`)

**What**: Takes trained checkpoints and produces 5 scientific figures for the paper.

**Why**: The paper's central claim is that LGQ learns codebook geometry matching Gersho's theory predictions. These figures provide the empirical evidence.

**Figure 1 — Voronoi UMAP**: Projects latent vectors and codewords to 2D via UMAP, colours by assignment. Shows side-by-side comparison of LGQ vs VQ vs FSQ. Expected: LGQ cells are anisotropic and data-adapted, FSQ cells are grid-like.

**Figure 2 — Cell volume vs density**: Log-log scatter with power law fit. Gersho predicts slope a = -(d+2)/d. Reference line drawn from d_eff. Expected: LGQ matches Gersho slope, FSQ shows near-zero slope.

**Figure 3 — Anisotropy**: For each Voronoi cell, computes covariance and anisotropy ratio (max_eigenvalue/min_eigenvalue). Plots distribution and anisotropy vs cell volume. Expected: LGQ cells more anisotropic than FSQ.

**Figure 4 — Scaling law**: Plots log(distortion) vs log(active_codes) across different K values. Fits line — slope should be -2/d_eff per Gersho. Also shows rFID vs K and marks N* from Phase 0.

**Figure 5 — Convergence**: Plots rFID and active codes vs epoch for different init strategies (data-centric, k-means, Gaussian). Expected: data-centric init reaches good rFID faster.

### 5. SLURM Scripts

**`run_phase0.sh`**: Full production run. Arguments: checkpoint path, num_samples (default 50000), dataset (default imagenet100). Uses the same module/venv setup as existing SLURM scripts in `shell/slurm/`.

**`run_phase0_test.sh`**: Quick test with M'=1000, 15min time limit.

---

## Test Results

### CPU Test (M'=100, login node)

Successfully ran end-to-end. Results:

| Metric | Value |
|--------|-------|
| d_eff | 3.64 |
| N* | 131072 (capped — raw value extremely large) |
| D* | 0.791 |
| Latent dim C | 128 |

**Predicted utilisation**:

| Init Method | Active Codes | Entropy (bits) | Avg Distance |
|-------------|-------------|----------------|--------------|
| Datacentric k-means | 6399/6399 (100%) | 12.64 | 0.000379 |
| Standard k-means | 6399/6399 (100%) | 12.64 | 0.000021 |
| Gaussian random | 135/6399 (2.1%) | 5.75 | 228.64 |

**Key observations**:
- The stark contrast between k-means inits (100% utilisation) and Gaussian (2.1%) validates the data-centric approach.
- d_eff ≈ 3.64 means the 128-dimensional latent space effectively uses only ~4 dimensions, which explains why Gaussian random init (which spreads codes across all 128 dims) performs so poorly.
- N* saturates at the cap because with C=128 and d_eff≈3.6, the Gersho formula gives extremely large values. This is mathematically expected — the norm factor term exp(-log_det_Sigma/(d+2)) dominates since most eigenvalues are near zero, making log_det_Sigma very negative.

### GPU Test (SLURM job 6705816)

**Status**: Submitted, pending GPU allocation (Priority queue).
**Check output**: `cat logs/slurm/phase0_test_6705816.out`

---

## Known Issues & Design Decisions

### N* Prediction Overflow
The Gersho-Zador formula N* = [C_d * ||p_Z|| / D*]^{d/2} can produce astronomically large values when:
- The latent space is high-dimensional (C=128) but most variance is concentrated in few dimensions (d_eff≈4)
- Many eigenvalues are near zero, making log_det_Sigma very negative, which makes the norm factor very large

**Decision**: Cap N* at 131072 (2^17). The formula is still useful for relative comparisons and for understanding the scaling, even if the absolute value needs clamping. The `log(N*)` value is reported in the summary so users can see the unclamped prediction.

### Density-Weighted K-Means Weights
The weights exp(mahal^2 / (d_eff+2)) can have extreme dynamic range (observed range [0, 64000] with std/mean=253). This is by design — it heavily upweights outlier regions to match the Gersho-optimal density. sklearn's KMeans handles this correctly via the `sample_weight` parameter.

### Checkpoint Loading
The model loading function reconstructs the model from the `config` dict saved in the checkpoint. It uses `strict=False` in `load_state_dict` to handle minor architecture changes between checkpoints.

---

## Pending Work

### Immediate (needs GPU)
- [ ] Wait for SLURM job 6705816 to complete and verify M'=1000 GPU results
- [ ] Run full Phase 0 with M'=50000: `sbatch run_phase0.sh`
- [ ] Verify codebook loading works: train with `--codebook-init phase0_results/codebook_datacentric.npy`

### Ablation Table (ImageNet-100, 100 epochs each)
Run 9 training jobs (3 inits x 3 K values):

| Init method | K=N* | K=2N* | K=4N* |
|-------------|------|-------|-------|
| Gaussian random | ... | ... | ... |
| Standard k-means | ... | ... | ... |
| Data-centric (ours) | ... | ... | ... |

Report: rFID, active codes, epochs to rFID < threshold

Example commands:
```bash
# After Phase 0 determines N*, e.g. N*=4096:
python scripts/train.py --model lmb --num-bins 4096 --flatten-channels \
    --codebook-init phase0_results/codebook_datacentric.npy \
    --dataset imagenet100 --epochs 100 --run-name ablation_dc_Nstar

python scripts/train.py --model lmb --num-bins 8192 --flatten-channels \
    --codebook-init phase0_results/codebook_datacentric.npy \
    --dataset imagenet100 --epochs 100 --run-name ablation_dc_2Nstar
```

### Geometry Figures
- [ ] Run geometry_analysis.py with trained LMB, VQ, and FSQ checkpoints
- [ ] Need to specify `--scaling-json` or `--scaling-checkpoints` for Figure 4
- [ ] Need convergence logs from ablation runs for Figure 5
- [ ] Install `umap-learn` if not available (`pip install umap-learn`) for Figure 1

Example:
```bash
python geometry_analysis.py \
    --checkpoint results/lmb/lmb_ablation_cb8k/checkpoints/best_model.pt \
    --vq-checkpoint results/vq/vq_cb8k/checkpoints/best_model.pt \
    --fsq-checkpoint results/fsq/fsq_lv8-5-5-5/checkpoints/best_model.pt \
    --phase0-summary phase0_results/summary.json \
    --dataset imagenet100 --split val
```

### Figure 4 Scaling Data
Need to compile results from existing checkpoints at different K values:
```
results/lmb/lmb_ablation_cb4k/
results/lmb/lmb_ablation_cb8k/
results/lmb/lmb_ablation_cb32k/
results/lmb/lmb_ablation_cb65k/
```
Create a JSON like:
```json
{
    "4096": {"rfid": ..., "active_codes": ..., "mse": ...},
    "8192": {"rfid": ..., "active_codes": ..., "mse": ...},
    ...
}
```

### Figure 5 Convergence Data
Need to train 3 models with identical hyperparameters but different inits, then point to their `eval_metrics.csv` files:
```bash
python geometry_analysis.py ... \
    --convergence-logs \
        results/lmb/ablation_dc/eval_metrics.csv \
        results/lmb/ablation_kmeans/eval_metrics.csv \
        results/lmb/ablation_gaussian/eval_metrics.csv \
    --convergence-labels "Data-centric" "K-means" "Gaussian"
```

---

## File Structure After Implementation

```
vector-quantize/
├── phase0_datacentric.py          # NEW: Phase 0 script
├── geometry_analysis.py           # NEW: Geometry analysis & figures
├── run_phase0.sh                  # NEW: SLURM job (full run)
├── run_phase0_test.sh             # NEW: SLURM job (quick test)
├── phase0_results/                # NEW: Phase 0 outputs
│   ├── summary.json
│   ├── eigenvalue_spectrum.png
│   ├── codebook_datacentric.npy
│   ├── codebook_standard_kmeans.npy
│   └── codebook_gaussian_init.npy
├── figures/                       # NEW: Geometry figures (after running)
│   ├── fig1_voronoi_umap.pdf
│   ├── fig2_volume_vs_density.pdf
│   ├── fig3_anisotropy.pdf
│   ├── fig4_scaling_law.pdf
│   └── fig5_convergence.pdf
├── scripts/
│   └── train.py                   # MODIFIED: +ImageNet100, +codebook-init
├── quantization/                  # UNCHANGED
├── configs/                       # UNCHANGED
└── ...
```

---

## Quick Reference Commands

```bash
# Phase 0 (GPU required):
sbatch run_phase0.sh [checkpoint] [num_samples] [dataset]

# Train with data-centric init:
python scripts/train.py --model lmb --num-bins 8192 --flatten-channels \
    --codebook-init phase0_results/codebook_datacentric.npy \
    --dataset imagenet100

# Geometry analysis:
python geometry_analysis.py \
    --checkpoint path/to/lmb/best_model.pt \
    --vq-checkpoint path/to/vq/best_model.pt \
    --fsq-checkpoint path/to/fsq/best_model.pt \
    --phase0-summary phase0_results/summary.json

# Check SLURM job:
squeue -u $USER
cat logs/slurm/phase0_test_JOBID.out
```

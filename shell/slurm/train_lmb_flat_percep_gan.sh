#!/bin/bash
# LMB flattened (single vector codebook) K=4096 with perceptual + PatchGAN losses.
#
#SBATCH --job-name=lmb_pg4k_flat
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"
SEED="${SEED:-1234}"
TAG="lmb_flat_cb4096_256_perceptgan_s${SEED}"

echo "=== $TAG ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

python3 scripts/train.py \
    --model lmb \
    --data-root "$DATA_ROOT" \
    --image-size 256 \
    --batch-size 8 \
    --grad-accum-steps 4 \
    --amp \
    --epochs 100 \
    --lr 3e-4 \
    --dim 256 \
    --embedding-dim 64 \
    --flatten-channels \
    --num-bins 4096 \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --perceptual-weight 1.0 \
    --gan-weight 0.1 \
    --disc-start 10000 \
    --num-workers 0 \
    --seed "$SEED" \
    --run-name "$TAG" \
    --keep-only-best-latest \
    --gpu 0

echo "=== $TAG done ==="

#!/bin/bash
# Ablation 3: squared L2 + random init + tau=64→6.4 (isolates tau scaling effect)
#
#SBATCH --job-name=lmb_abl3
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CODEBOOK_SIZE="${CODEBOOK_SIZE:-16384}"
DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"
SEED="${SEED:-1234}"

TAG="lmb_abl3_sq_random_tau64_cb${CODEBOOK_SIZE}_s${SEED}"
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
    --num-bins "$CODEBOOK_SIZE" \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --distance-type l2_sq \
    --tau-start 64.0 \
    --tau-end 6.4 \
    --flatten-channels \
    --init-method random \
    --perceptual-weight 1.0 \
    --gan-weight 0.1 \
    --disc-start 10000 \
    --wandb \
    --num-workers 0 \
    --seed "$SEED" \
    --run-name "$TAG" \
    --keep-only-best-latest \
    --gpu 0

echo "=== $TAG done ==="

#!/bin/bash
# VQ – standard Vector Quantization with EMA updates.
# DDP on 4xH100 per hopper node. Submitted by submit_all.sh with CODEBOOK_SIZE set in the environment.
#
#SBATCH --job-name=vq
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=2-00:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

CODEBOOK_SIZE="${CODEBOOK_SIZE:-16384}"
DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"
SEED="${SEED:-1234}"
RESUME="${RESUME:-latest}"

NGPUS=4
PER_GPU_BATCH=8
GLOBAL_BATCH=$((PER_GPU_BATCH * NGPUS))
LR=3e-4

TAG="vq_cb${CODEBOOK_SIZE}_256_ddp${NGPUS}_gb${GLOBAL_BATCH}_s${SEED}"
echo "=== $TAG ==="
echo "Node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES  global_batch=$GLOBAL_BATCH"

MASTER_PORT=$((20000 + RANDOM % 20000))


srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
        --model vq \
        --data-root "$DATA_ROOT" \
        --dataset imagenet \
        --image-size 256 \
        --batch-size $PER_GPU_BATCH \
        --grad-accum-steps 1 \
        --amp \
        --epochs 100 \
        --lr $LR \
        --dim 256 \
        --embedding-dim 64 \
        --codebook-size "$CODEBOOK_SIZE" \
        --commitment-weight 1.0 \
        --decay 0.8 \
        --perceptual-weight 1.0 \
        --gan-weight 0.1 \
        --disc-start 10000 \
        --wandb \
        --num-workers 8 \
        --seed "$SEED" \
        --run-name "$TAG" \
        --keep-only-best-latest \
        ${RESUME:+--resume "$RESUME"}

echo "=== $TAG done ==="

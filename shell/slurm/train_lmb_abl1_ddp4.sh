#!/bin/bash
# abl1 recipe on 4 GPUs — DDP used purely for throughput (4x), not for batch scaling.
# Per-GPU batch 8, grad_accum 1, 4 ranks -> effective batch 32, identical to single-GPU abl1
# (which used batch 8 * grad_accum 4 = 32). LR, tau schedule, disc_start all unchanged.
#
#SBATCH --job-name=lmb_abl1_ddp4
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

# Match abl1 exactly: effective batch = PER_GPU_BATCH * NGPUS * GRAD_ACCUM = 8 * 4 * 1 = 32
NGPUS=4
PER_GPU_BATCH=8
GRAD_ACCUM=1

TAG="lmb_abl1_ddp4_cb${CODEBOOK_SIZE}_s${SEED}${TAG_SUFFIX:+_${TAG_SUFFIX}}"
echo "=== $TAG ==="
echo "Node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES  effective_batch=$((PER_GPU_BATCH * NGPUS * GRAD_ACCUM))"

MASTER_PORT=$((20000 + RANDOM % 20000))


srun torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NGPUS \
    --master_port=$MASTER_PORT \
    scripts/train.py \
        --model lmb \
        --data-root "$DATA_ROOT" \
        --dataset imagenet \
        --image-size 256 \
        --batch-size $PER_GPU_BATCH \
        --grad-accum-steps $GRAD_ACCUM \
        --amp \
        --epochs 100 \
        --lr 3e-4 \
        --dim 256 \
        --embedding-dim 64 \
        --num-bins "$CODEBOOK_SIZE" \
        --lambda-peak 0.005 \
        --lambda-bins 0.005 \
        --distance-type l2_sq \
        --tau-start 1.0 \
        --tau-end 0.1 \
        --flatten-channels \
        --init-method random \
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

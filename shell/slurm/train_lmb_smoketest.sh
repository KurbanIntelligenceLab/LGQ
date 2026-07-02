#!/bin/bash
# DDP smoke test: 4-GPU LMB on hopper, 20 steps, 1 epoch, no eval/wandb.
# Use this to verify the DDP path end-to-end before submitting the full 100-epoch run.
#
#SBATCH --job-name=lmb_smoke
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=400G
#SBATCH --time=00:15:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
# Surface NCCL errors early if something is wrong with inter-GPU comms.
export NCCL_DEBUG=WARN
export TORCH_NCCL_BLOCKING_WAIT=1

DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-16384}"

NGPUS=4
PER_GPU_BATCH=32
TAG="lmb_smoke_ddp${NGPUS}_gb$((PER_GPU_BATCH * NGPUS))"
echo "=== $TAG ==="
echo "Node: $(hostname)  GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

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
        --grad-accum-steps 1 \
        --amp \
        --epochs 1 \
        --max-steps 5 \
        --lr 8e-4 \
        --warmup-steps 3 \
        --lr-schedule cosine \
        --min-lr 1e-6 \
        --dim 256 \
        --embedding-dim 64 \
        --num-bins "$CODEBOOK_SIZE" \
        --lambda-peak 0.005 \
        --lambda-bins 0.005 \
        --tau-start 1.0 \
        --tau-end 0.1 \
        --flatten-channels \
        --init-method random \
        --perceptual-weight 1.0 \
        --gan-weight 0.1 \
        --disc-start 5 \
        --num-workers 4 \
        --no-eval \
        --seed 1234 \
        --run-name "$TAG" \
        --keep-only-best-latest

echo "=== $TAG done ==="

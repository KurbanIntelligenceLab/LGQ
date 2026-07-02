#!/bin/bash
# Generic 128x128 codebook ablation script for all quantizer types.
# Submitted with MODEL, CODEBOOK_SIZE, RUN_NAME, and model-specific args via --export.
#
#SBATCH --job-name=abl
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

MODEL="${MODEL:?MODEL must be set}"
RUN_NAME="${RUN_NAME:?RUN_NAME must be set}"
DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== ${RUN_NAME} (model=${MODEL}) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

python3 scripts/train.py \
    --model "$MODEL" \
    --data-root "$DATA_ROOT" \
    --image-size 128 \
    --batch-size 32 \
    --epochs 100 \
    --lr 3e-4 \
    --dim 128 \
    --embedding-dim 128 \
    --num-workers 4 \
    --seed 1234 \
    --run-name "$RUN_NAME" \
    --keep-only-best-latest \
    --resume latest \
    --gpu 0 \
    $EXTRA_ARGS

echo "=== ${RUN_NAME} done ==="

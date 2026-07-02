#!/bin/bash
# Resume LMB codebook ablation experiments.
# cb4k: restart from scratch (codebook collapsed)
# cb8k, cb32k, cb65k: resume from latest checkpoint
#
#SBATCH --job-name=lmb_abl
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

# Injected via --export
NUM_BINS="${NUM_BINS:-8192}"
RESUME="${RESUME:-latest}"
RUN_NAME="${RUN_NAME:-lmb_ablation_cb8k}"
DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"

echo "=== ${RUN_NAME} (bins=${NUM_BINS}, resume=${RESUME}) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

python3 scripts/train.py \
    --model lmb \
    --data-root "$DATA_ROOT" \
    --image-size 128 \
    --batch-size 32 \
    --epochs 100 \
    --lr 3e-4 \
    --dim 128 \
    --embedding-dim 128 \
    --num-bins "$NUM_BINS" \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --flatten-channels \
    --init-method random \
    --num-workers 4 \
    --seed 1234 \
    --run-name "$RUN_NAME" \
    --keep-only-best-latest \
    --resume "$RESUME" \
    --gpu 0

echo "=== ${RUN_NAME} done ==="

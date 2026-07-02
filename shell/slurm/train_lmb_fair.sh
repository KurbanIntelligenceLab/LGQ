#!/bin/bash
# LMB (LGQ) – per-channel fair mode (FSQ-style structure).
# Submitted by submit_all.sh with CODEBOOK_SIZE set in the environment.
#
# Codebook size -> levels mapping:
#   4096  -> 8 8 8 8
#   8192  -> 16 8 8 8  (actually 8192 = 16*8*8*8 = 8192? No: 16*8*8*8=8192. Hmm wait.
#            Actually: 8*8*8*8=4096, so for 8192 we need e.g. 16*8*8*8=8192?
#            Let me recalculate: 16*8*8*8 = 8192. Yes.)
#   16384 -> 16 16 8 8
#   32768 -> 16 16 16 8
#   65536 -> 16 16 16 16
#
#SBATCH --job-name=lmb_fair
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

# Injected by submit_all.sh via --export
CODEBOOK_SIZE="${CODEBOOK_SIZE:-16384}"
DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"
SEED="${SEED:-1234}"

# Map codebook size to per-channel levels
case "$CODEBOOK_SIZE" in
    4096)  LEVELS="8 8 8 8" ;;
    8192)  LEVELS="16 8 8 8" ;;
    16384) LEVELS="16 16 8 8" ;;
    32768) LEVELS="16 16 16 8" ;;
    65536) LEVELS="16 16 16 16" ;;
    *)     echo "ERROR: unsupported codebook size $CODEBOOK_SIZE for fair mode"; exit 1 ;;
esac

TAG="lmb_fair_cb${CODEBOOK_SIZE}_256_s${SEED}"
echo "=== $TAG  levels=($LEVELS) ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"

python3 scripts/train.py \
    --model lmb \
    --data-root "$DATA_ROOT" \
    --image-size 256 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --amp \
    --epochs 100 \
    --lr 3e-4 \
    --dim 256 \
    --embedding-dim 64 \
    --perchannel-fair \
    --lmb-levels $LEVELS \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --num-workers 0 \
    --seed "$SEED" \
    --run-name "$TAG" \
    --keep-only-best-latest \
    --gpu 0

echo "=== $TAG done ==="

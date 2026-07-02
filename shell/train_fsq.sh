#!/bin/bash
# Train Finite Scalar Quantization (FSQ) model on ImageNet
# Fair comparison: uses same shared hyperparameters as other models

set -e

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Add project root to PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

# ============================================================================
# SHARED HYPERPARAMETERS (same across all models for fair comparison)
# ============================================================================
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
DIM="${DIM:-128}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
GPU="${GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1234}"

# ============================================================================
# MODEL-SPECIFIC HYPERPARAMETERS
# ============================================================================
# FSQ levels: product = codebook size (8*8*8*8*4 = 16384 codes)
LEVELS="${LEVELS:-8 8 8 8 4}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --levels) LEVELS="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --embedding-dim) EMBEDDING_DIM="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --run-name) RUN_NAME="--run-name $2"; shift 2 ;;
        --fid-samples) FID_SAMPLES="--fid-samples $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Training FSQ model on ImageNet"
echo "=============================="
echo "Shared params: batch_size=$BATCH_SIZE, lr=$LR, dim=$DIM, emb_dim=$EMBEDDING_DIM"
echo "FSQ levels: $LEVELS"
echo "GPU: $GPU"
echo ""

python3 scripts/train.py \
    --model fsq \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --levels $LEVELS \
    --dim "$DIM" \
    --embedding-dim "$EMBEDDING_DIM" \
    --gpu "$GPU" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    ${RUN_NAME:-} \
    ${FID_SAMPLES:-}

echo ""
echo "Training completed!"

#!/bin/bash
# Train Rotation-trick VQ (rot_vq) on ImageNet
# Uses the rotation trick from rotation_trick-main (Fifty et al., ICLR 2025)
# with our configs and backbone for the main experiment (fair comparison).

set -e

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Add project root to PYTHONPATH
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

# ============================================================================
# SHARED HYPERPARAMETERS (same across all models for fair comparison)
# ============================================================================
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
DIM="${DIM:-128}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
GPU="${GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1234}"

# ============================================================================
# MODEL-SPECIFIC HYPERPARAMETERS (same as main VQ experiment)
# ============================================================================
CODEBOOK_SIZE="${CODEBOOK_SIZE:-16384}"
COMMITMENT_WEIGHT="${COMMITMENT_WEIGHT:-1.0}"
DECAY="${DECAY:-0.8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --image-size) IMAGE_SIZE="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --codebook-size) CODEBOOK_SIZE="$2"; shift 2 ;;
        --dim) DIM="$2"; shift 2 ;;
        --embedding-dim) EMBEDDING_DIM="$2"; shift 2 ;;
        --commitment-weight) COMMITMENT_WEIGHT="$2"; shift 2 ;;
        --decay) DECAY="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --run-name) RUN_NAME="--run-name $2"; shift 2 ;;
        --fid-samples) FID_SAMPLES="--fid-samples $2"; shift 2 ;;
        --resume) RESUME="--resume $2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Training Rotation-trick VQ (rot_vq) on ImageNet"
echo "==============================================="
echo "Shared params: image_size=$IMAGE_SIZE, batch_size=$BATCH_SIZE, lr=$LR, dim=$DIM, emb_dim=$EMBEDDING_DIM"
echo "Codebook size: $CODEBOOK_SIZE (rotation trick always on)"
echo "GPU: $GPU"
echo ""

python3 scripts/train.py \
    --model rot_vq \
    --data-root "$DATA_ROOT" \
    --image-size "$IMAGE_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --codebook-size "$CODEBOOK_SIZE" \
    --dim "$DIM" \
    --embedding-dim "$EMBEDDING_DIM" \
    --commitment-weight "$COMMITMENT_WEIGHT" \
    --decay "$DECAY" \
    --gpu "$GPU" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    ${RUN_NAME:-} \
    ${FID_SAMPLES:-} \
    ${RESUME:-}

echo ""
echo "Training completed!"

#!/bin/bash
# Train LMB-VAE with flattened channels (as suggested in email)
# This flattens channels before normalization and uses a single codebook

set -e

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

# Configuration matching the email suggestion
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-3e-4}"
DIM="${DIM:-128}"
EMBEDDING_DIM="${EMBEDDING_DIM:-128}"
NUM_BINS="${NUM_BINS:-16384}"
GPU="${GPU:-1}"

echo "Training LMB-VAE with FLATTENED CHANNELS"
echo "========================================"
echo "This implements: flatten channels before normalization, use single codebook"
echo "Configuration:"
echo "  - embedding_dim: $EMBEDDING_DIM"
echo "  - num_bins: $NUM_BINS"
echo "  - flatten_channels: True"
echo "  - batch_size: $BATCH_SIZE"
echo "  - lr: $LR"
echo "  - epochs: $EPOCHS"
echo ""

python3 scripts/train.py \
    --model lmb \
    --data-root "$DATA_ROOT" \
    --batch-size "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --num-bins "$NUM_BINS" \
    --dim "$DIM" \
    --embedding-dim "$EMBEDDING_DIM" \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --flatten-channels \
    --gpu "$GPU" \
    --num-workers 8 \
    --seed 1234

echo ""
echo "Training completed!"

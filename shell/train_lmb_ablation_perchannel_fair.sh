#!/bin/bash
# LMB Ablation: Per-channel fair (same structure as FSQ: 4 channels, [16,16,8,8] -> 16K)

set -e

cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
mkdir -p logs

/venv/encoder_decoder/bin/python3 scripts/train.py \
    --model lmb \
    --data-root "${DATA_ROOT:-data/imagenet}" \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0003 \
    --dim 128 \
    --embedding-dim 128 \
    --perchannel-fair \
    --lmb-levels 16 16 8 8 \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --init-method random \
    --image-size 128 \
    --gpu "${GPU:-0}" \
    --num-workers 8 \
    --seed 1234 \
    --run-name lmb_ablation_perchannel_fair \
    "$@"

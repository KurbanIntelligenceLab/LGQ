#!/bin/bash
# LMB Ablation: Medium Codebook (16K = 16384 bins, ~14 bits) - Baseline

set -e

cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

/venv/encoder_decoder/bin/python3 scripts/train.py \
    --model lmb \
    --data-root data/imagenet \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0003 \
    --num-bins 16384 \
    --dim 128 \
    --embedding-dim 128 \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --image-size 128 \
    --gpu 1 \
    --num-workers 8 \
    --seed 1234 \
    --flatten-channels \
    --run-name lmb_ablation_cb16k > logs/lmb_ablation_cb16k.log 2>&1 &

echo "LMB Codebook 16K ablation experiment started on GPU 1! PID: $!"

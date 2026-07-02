#!/bin/bash
# LMB Ablation: Large Codebook (65K = 65536 bins, ~16 bits)

set -e

cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

/venv/encoder_decoder/bin/python3 scripts/train.py \
    --model lmb \
    --data-root data/imagenet \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0003 \
    --num-bins 65536 \
    --dim 128 \
    --embedding-dim 128 \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --image-size 128 \
    --gpu 2 \
    --num-workers 8 \
    --seed 1234 \
    --flatten-channels \
    --run-name lmb_ablation_cb65k > logs/lmb_ablation_cb65k.log 2>&1 &

echo "LMB Codebook 65K ablation experiment started on GPU 2! PID: $!"

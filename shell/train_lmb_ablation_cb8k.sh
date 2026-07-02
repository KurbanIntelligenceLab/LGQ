#!/bin/bash
# LMB Ablation: Medium-Small Codebook (8K = 8192 bins, ~13 bits)

set -e

cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

/venv/encoder_decoder/bin/python3 scripts/train.py \
    --model lmb \
    --data-root data/imagenet \
    --batch-size 32 \
    --epochs 100 \
    --lr 0.0003 \
    --num-bins 8192 \
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
    --run-name lmb_ablation_cb8k > logs/lmb_ablation_cb8k.log 2>&1 &

echo "LMB Codebook 8K ablation experiment started on GPU 1! PID: $!"

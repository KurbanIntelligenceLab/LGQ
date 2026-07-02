#!/bin/bash
# LMB Ablation: Regularization Grid - Peak 0.002, Bins 0.002

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
    --lambda-peak 0.002 \
    --lambda-bins 0.002 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --image-size 128 \
    --gpu 1 \
    --num-workers 8 \
    --seed 1234 \
    --flatten-channels \
    --run-name lmb_ablation_reg_grid_peak002_bins002 > logs/lmb_ablation_reg_grid_peak002_bins002.log 2>&1 &

echo "LMB Regularization Grid (peak=0.002, bins=0.002) experiment started on GPU 1! PID: $!"

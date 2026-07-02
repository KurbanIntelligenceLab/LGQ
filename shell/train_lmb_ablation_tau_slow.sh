#!/bin/bash
# LMB Ablation: Slow Temperature Annealing

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
    --tau-end 0.2 \
    --image-size 128 \
    --gpu 0 \
    --num-workers 8 \
    --seed 1234 \
    --flatten-channels \
    --run-name lmb_ablation_tau_slow > logs/lmb_ablation_tau_slow.log 2>&1 &

echo "LMB Slow Temperature Annealing ablation experiment started on GPU 0! PID: $!"

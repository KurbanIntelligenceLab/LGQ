#!/bin/bash
# LMB Ablation: Test lambda_floor and p_min to improve utilization
# Baseline params + floor/min regularization
# p_min should be HIGHER than current soft perplexity to encourage higher utilization

set -e

cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

# Experiment: lambda_floor with p_min set to encourage higher perplexity
# Current hard perplexity ~5806-7143, soft perplexity is typically lower
# Set p_min=6000 to push soft perplexity higher (encourages more uniform usage)
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
    --lambda-floor 0.01 \
    --p-min 6500 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --image-size 128 \
    --gpu 1 \
    --num-workers 8 \
    --seed 1234 \
    --flatten-channels \
    --run-name lmb_ablation_floor_min > logs/lmb_ablation_floor_min.log 2>&1 &

echo "LMB Floor/Min regularization experiment started on GPU 1! PID: $!"
echo "  p_min=6500 (target soft perplexity, should be > current soft perplexity)"
echo "  lambda_floor=0.01 (weight for perplexity floor loss)"

#!/bin/bash
# LFQ codebook ablation: 32K. GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model lfq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --codebook-size 32768 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name lfq_cb32k >> logs/lfq_cb32k.log 2>&1
echo "LFQ cb32k finished on GPU 1."

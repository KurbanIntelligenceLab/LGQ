#!/bin/bash
# VQ codebook ablation: 32K. GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model vq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --codebook-size 32768 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name vq_cb32k >> logs/vq_cb32k.log 2>&1
echo "VQ cb32k finished on GPU 1."

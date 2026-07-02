#!/bin/bash
# LFQ codebook ablation: 65K (match LMB cb65k). GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model lfq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --codebook-size 65536 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name lfq_cb65k >> logs/lfq_cb65k.log 2>&1
echo "LFQ cb65k finished on GPU 1."

#!/bin/bash
# FSQ codebook ablation: 4K (levels 8 8 8 8 = 4096). GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model fsq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --levels 8 8 8 8 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name fsq_cb4k >> logs/fsq_cb4k.log 2>&1
echo "FSQ cb4k finished on GPU 1."

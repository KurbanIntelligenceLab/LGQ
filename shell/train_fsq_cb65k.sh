#!/bin/bash
# FSQ codebook ablation: 65K (levels 64 16 8 8 = 65536). GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model fsq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --levels 64 16 8 8 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name fsq_cb65k >> logs/fsq_cb65k.log 2>&1
echo "FSQ cb65k finished on GPU 1."

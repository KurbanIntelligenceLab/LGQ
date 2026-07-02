#!/bin/bash
# FSQ codebook ablation: 8K (levels 16 8 8 8 = 8192). GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model fsq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --levels 16 8 8 8 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name fsq_cb8k >> logs/fsq_cb8k.log 2>&1
echo "FSQ cb8k finished on GPU 1."

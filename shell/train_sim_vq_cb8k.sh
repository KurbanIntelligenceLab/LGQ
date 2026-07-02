#!/bin/bash
# SimVQ codebook ablation: 8K. GPU 1.
set -e
cd /workspace/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
/venv/encoder_decoder/bin/python3 scripts/train.py --model sim_vq --data-root data/imagenet --batch-size 32 --epochs 100 --lr 0.0003 --codebook-size 8192 --dim 128 --embedding-dim 128 --image-size 128 --gpu 1 --num-workers 8 --seed 1234 --run-name sim_vq_cb8k >> logs/sim_vq_cb8k.log 2>&1
echo "SimVQ cb8k finished on GPU 1."

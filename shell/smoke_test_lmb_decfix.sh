#!/bin/bash
# Smoke test: LMB with decoder fix (base_ch*2 internal width)
# Quick 50-step run to verify training works end-to-end
#
#SBATCH --job-name=smoke_lmb_decfix
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/smoke_lmb_decfix_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/smoke_lmb_decfix_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

echo "=== Smoke Test: LMB with Decoder Fix ==="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(python3 --version)"
echo ""

python3 scripts/train.py --model lmb \
    --data-root /N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet \
    --dataset imagenet100 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --epochs 2 \
    --lr 3e-4 \
    --seed 1234 \
    --dim 256 \
    --embedding-dim 64 \
    --image-size 256 \
    --compression-factor 16 \
    --num-residual-layers 2 \
    --keep-only-best-latest \
    --csv-log-every 10 \
    --print-every 10 \
    --max-steps 50 \
    --amp \
    --num-workers 0 \
    --num-bins 8192 \
    --flatten-channels \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --run-name smoke_lmb_decfix

echo ""
echo "=== Smoke test complete ==="

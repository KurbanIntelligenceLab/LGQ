#!/bin/bash
# Generic rFID eval — point at an arbitrary checkpoint via env vars:
#   CKPT=path/to/foo.pt OUT_CSV=path/to/result.csv sbatch eval_rfid_explicit.sh
#
#SBATCH --job-name=eval_rfid_explicit
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --exclude=g36
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
: "${CKPT:?Set CKPT=<path/to/checkpoint.pt>}"
: "${OUT_CSV:?Set OUT_CSV=<path/to/out.csv>}"
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export TORCH_CUDNN_V8_API_DISABLED=1

DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"

echo "=== rFID eval: $CKPT ==="
nvidia-smi -L || true

python scripts/evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --batch-size 32 \
    --num-workers 0 \
    --fid-samples 10000 \
    --write-eval-csv "$OUT_CSV"

echo "=== done: $OUT_CSV ==="

#!/bin/bash
# One-off: rFID for the cb16384 LMB snapshot taken on 2026-04-26.
#
#SBATCH --job-name=rfid_cb16384
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
CKPT="results/lmb/_rfid_snapshots_2026-04-26/cb16384_latest.pt"
OUT_CSV="results/lmb/_rfid_snapshots_2026-04-26/cb16384_rfid.csv"

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

echo "=== done ==="

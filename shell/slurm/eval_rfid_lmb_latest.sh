#!/bin/bash
# rFID eval against the LATEST checkpoint of each lmb_abl1_ddp4_cb${K}_s1234 run.
# Snapshots live in results/lmb/_rfid_snapshots_2026-04-28/ (symlinks → latest_model.pt).
# Pass codebook size via env: K=16384 sbatch ... eval_rfid_lmb_latest.sh
#
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
: "${K:?Set K=<codebook_size>}"
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

export TORCH_CUDNN_V8_API_DISABLED=1

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
SNAP_DIR="results/lmb/_rfid_snapshots_2026-04-28"
CKPT="${SNAP_DIR}/cb${K}_latest.pt"
OUT_CSV="${SNAP_DIR}/cb${K}_rfid.csv"

echo "=== rFID eval K=${K}: $CKPT ==="
nvidia-smi -L || true

python scripts/evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --batch-size 32 \
    --num-workers 0 \
    --fid-samples 10000 \
    --write-eval-csv "$OUT_CSV"

echo "=== done K=${K} ==="

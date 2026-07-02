#!/bin/bash
# srun smoke: tiny rFID run to verify the eval path works.
set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
CKPT="results/lmb/_rfid_snapshots_2026-04-26/cb16384_latest.pt"
OUT_CSV="results/lmb/_rfid_snapshots_2026-04-26/cb16384_rfid_smoke.csv"

echo "=== smoke rFID: $CKPT ==="
nvidia-smi -L || true

python scripts/evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --batch-size 16 \
    --num-workers 0 \
    --num-samples 256 \
    --fid-samples 256 \
    --write-eval-csv "$OUT_CSV"

echo "=== smoke done ==="

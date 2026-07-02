#!/bin/bash
# Post-training rFID eval helper. Call from a training launcher's tail:
#   bash shell/slurm/_post_train_rfid_eval.sh <run_dir>
# where <run_dir> contains checkpoints/latest_model.pt.
# Writes rfid_eval.csv into <run_dir>/.
#
# Skips silently (exit 0) if no checkpoint exists, so a failed/preempted
# training run does not poison the launcher's exit code under `set -e`.

set -u
RUN_DIR="${1:?Usage: $0 <run_dir>}"
CKPT="${RUN_DIR}/checkpoints/latest_model.pt"
OUT_CSV="${RUN_DIR}/rfid_eval.csv"
DATA_ROOT="${DATA_ROOT:-/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet}"

if [ ! -f "$CKPT" ]; then
    echo "[post-train rFID] no checkpoint at $CKPT; skipping"
    exit 0
fi

echo "[post-train rFID] Evaluating $CKPT"
python scripts/evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --batch-size 32 \
    --num-workers 0 \
    --fid-samples 10000 \
    --write-eval-csv "$OUT_CSV"
echo "[post-train rFID] done -> $OUT_CSV"

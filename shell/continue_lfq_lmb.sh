#!/bin/bash
# Restart LFQ and LMB main with --resume latest (continue from last epoch).
# Stops the current lfq_cb16k_main (running from scratch) and switches to
# the LFQ run used in model comparison. Starts LMB main if not running.

set -e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"

# Kill current LFQ (lfq_cb16k_main - runs from scratch, not resuming)
LFQ_PIDS=$(pgrep -f "train.py.*lfq.*lfq_cb16k_main" 2>/dev/null || true)
if [[ -n "$LFQ_PIDS" ]]; then
  echo "Stopping LFQ lfq_cb16k_main (PID $LFQ_PIDS)..."
  kill $LFQ_PIDS 2>/dev/null || true
  sleep 2
fi

mkdir -p logs

# LFQ: resume lfq_cb16384_bs32... (main run in model comparison, ~49 epochs)
LFQ_DIR="results/lfq/lfq_cb16384_bs32_lr3e-4_dim128_20260123_221044_6f79"
if [[ -d "$LFQ_DIR" ]] && [[ -f "$LFQ_DIR/checkpoints/latest_model.pt" ]]; then
  echo "Starting LFQ with --resume latest (output: $LFQ_DIR)..."
  nohup python3 scripts/train.py \
    --model lfq --data-root "$DATA_ROOT" --resume latest --output-dir "$LFQ_DIR" \
    --codebook-size 16384 --batch-size 32 --epochs 100 --lr 0.0003 \
    --dim 128 --embedding-dim 128 --image-size 128 \
    --gpu 3 --num-workers 8 --seed 1234 \
    > logs/lfq_continue.log 2>&1 &
  echo "  LFQ on GPU 3 (PID $!)"
else
  echo "  SKIP LFQ: $LFQ_DIR or checkpoint not found"
fi

# LMB main: resume lmb_nb16384... (main LMB in model comparison, ~42 epochs)
LMB_DIR="results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd"
if [[ -d "$LMB_DIR" ]] && [[ -f "$LMB_DIR/checkpoints/latest_model.pt" ]]; then
  echo "Starting LMB main with --resume latest (output: $LMB_DIR)..."
  nohup python3 scripts/train.py \
    --model lmb --data-root "$DATA_ROOT" --resume latest --output-dir "$LMB_DIR" \
    --num-bins 16384 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 \
    --lambda-floor 0.0 --p-min 0.0 --tau-start 1.0 --tau-end 0.1 \
    --batch-size 32 --epochs 100 --lr 0.0003 \
    --dim 128 --embedding-dim 128 --image-size 128 \
    --gpu 2 --num-workers 8 --seed 1234 \
    > logs/lmb_main_continue.log 2>&1 &
  echo "  LMB main on GPU 2 (PID $!)"
else
  echo "  SKIP LMB: $LMB_DIR or checkpoint not found"
fi

echo ""
echo "Done. Logs: logs/lfq_continue.log, logs/lmb_main_continue.log"
disown

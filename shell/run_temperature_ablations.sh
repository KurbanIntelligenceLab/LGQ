#!/bin/bash
# Run LMB temperature ablations: tau_fast (0.05), tau_slow (0.2), tau_fixed (1.0)
# Resumes from latest if checkpoint exists. Uses GPU 1 (or set GPU env).

set -e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
GPU="${GPU:-1}"

echo "Launching LMB temperature ablations (--resume latest) on GPU $GPU..."
echo "=========================================================================="

for run_name in lmb_ablation_tau_fast lmb_ablation_tau_slow lmb_ablation_tau_fixed; do
  out_dir="results/lmb/${run_name}"
  mkdir -p "$out_dir"
  case "$run_name" in
    lmb_ablation_tau_fast)  tau_end=0.05 ;;
    lmb_ablation_tau_slow)  tau_end=0.2  ;;
    lmb_ablation_tau_fixed) tau_end=1.0 ;;
    *) tau_end=0.1 ;;
  esac
  nohup python3 scripts/train.py \
    --model lmb --data-root "$DATA_ROOT" --resume latest --output-dir "$out_dir" \
    --num-bins 16384 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 \
    --lambda-floor 0.0 --p-min 0.0 --tau-start 1.0 --tau-end "$tau_end" \
    --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 \
    --image-size 128 --gpu "$GPU" --num-workers 8 --seed 1234 --run-name "$run_name" \
    --keep-only-best-latest \
    > /dev/null 2>&1 &
  echo "  $run_name (tau_end=$tau_end) on GPU $GPU (PID $!)"
  sleep 2
done

echo ""
echo "All 3 temperature ablations launched!"
echo "Logs disabled (output to /dev/null)"
disown

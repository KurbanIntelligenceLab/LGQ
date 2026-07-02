#!/bin/bash
# Run only the 5 basic LMB regularization ablations: none, weak, strong, bins_only, peak_only.
# No grid runs. Resumes from latest if checkpoint exists. Uses GPUs 0-3 (5th job shares GPU 0).
#
# Time: 5 runs × (same time as one LMB run). With 4 GPUs, 4 run in parallel + 1 after
#       → wall time ~1.2–1.5× a single LMB run. Safe to run without slowing other experiments much.

set -e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"

run_reg() {
  local gpu=$1
  local run_name=$2
  local lp=$3
  local lb=$4
  local out_dir="results/lmb/${run_name}"
  mkdir -p "$out_dir"
  nohup python3 scripts/train.py \
    --model lmb --data-root "$DATA_ROOT" --resume latest --output-dir "$out_dir" \
    --num-bins 16384 --flatten-channels --lambda-peak "$lp" --lambda-bins "$lb" \
    --lambda-floor 0.0 --p-min 0.0 --tau-start 1.0 --tau-end 0.1 \
    --batch-size 32 --epochs 10 --lr 0.0003 --dim 128 --embedding-dim 128 \
    --image-size 128 --gpu "$gpu" --num-workers 8 --seed 1234 --run-name "$run_name" \
    --keep-only-best-latest \
    > /dev/null 2>&1 &
  echo "  $run_name on GPU $gpu (PID $!)"
}

echo "Launching 5 basic LMB regularization ablations (none, weak, strong, bins_only, peak_only)..."
echo "=========================================================================="

run_reg 0 lmb_ablation_reg_none 0.0 0.0
run_reg 1 lmb_ablation_reg_weak 0.002 0.002
run_reg 2 lmb_ablation_reg_strong 0.01 0.01
run_reg 3 lmb_ablation_reg_bins_only 0.0 0.01
run_reg 0 lmb_ablation_reg_peak_only 0.01 0.0

echo ""
echo "All 5 basic LMB reg ablations launched!"
echo "Logs disabled (output to /dev/null)"
disown

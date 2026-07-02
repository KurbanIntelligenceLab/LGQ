#!/bin/bash
# Run LMB regularization ablations: none, weak, strong, bins_only, peak_only, grid (various lambda_peak x lambda_bins)
# Resumes from latest if checkpoint exists. Distributes across GPUs 0-3.

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
    --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 \
    --image-size 128 --gpu "$gpu" --num-workers 8 --seed 1234 --run-name "$run_name" \
    --keep-only-best-latest \
    > /dev/null 2>&1 &
  echo "  $run_name on GPU $gpu (PID $!)"
}

echo "Launching LMB regularization ablations (--resume latest)..."
echo "=========================================================================="

# Basic reg ablations
run_reg 0 lmb_ablation_reg_none 0.0 0.0
run_reg 1 lmb_ablation_reg_weak 0.002 0.002
run_reg 2 lmb_ablation_reg_strong 0.01 0.01
run_reg 3 lmb_ablation_reg_bins_only 0.0 0.01
run_reg 0 lmb_ablation_reg_peak_only 0.01 0.0

sleep 2

# Grid: peak 0.002
run_reg 1 lmb_ablation_reg_grid_peak002_bins002 0.002 0.002
run_reg 2 lmb_ablation_reg_grid_peak002_bins005 0.002 0.005
run_reg 3 lmb_ablation_reg_grid_peak002_bins010 0.002 0.01

sleep 2

# Grid: peak 0.005
run_reg 0 lmb_ablation_reg_grid_peak005_bins002 0.005 0.002
run_reg 1 lmb_ablation_reg_grid_peak005_bins010 0.005 0.01

sleep 2

# Grid: peak 0.01
run_reg 2 lmb_ablation_reg_grid_peak010_bins002 0.01 0.002
run_reg 3 lmb_ablation_reg_grid_peak010_bins005 0.01 0.005
run_reg 0 lmb_ablation_reg_grid_peak010_bins010 0.01 0.01

echo ""
echo "All 13 LMB reg ablations launched!"
echo "Logs disabled (output to /dev/null)"
disown

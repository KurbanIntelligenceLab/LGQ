#!/bin/bash
# Continue LMB reg ablations + SIM_VQ + LMB main. All use --resume latest.
# Distribution: GPU 0: 2 | GPU 1: 2 | GPU 2: 2 | GPU 3: 1

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
  if [[ ! -d "$out_dir" ]]; then
    echo "  SKIP $run_name: no output dir $out_dir"
    return
  fi
  mkdir -p logs
  nohup python3 scripts/train.py \
    --model lmb --data-root "$DATA_ROOT" --resume latest --output-dir "$out_dir" \
    --num-bins 16384 --flatten-channels --lambda-peak "$lp" --lambda-bins "$lb" \
    --lambda-floor 0.0 --p-min 0.0 --tau-start 1.0 --tau-end 0.1 \
    --batch-size 32 --epochs 25 --lr 0.0003 --dim 128 --embedding-dim 128 \
    --image-size 128 --gpu "$gpu" --num-workers 8 --seed 1234 --run-name "$run_name" \
    --keep-only-best-latest \
    > "logs/${run_name}.log" 2>&1 &
  echo "  $run_name on GPU $gpu (PID $!)"
}

run_main() {
  local gpu=$1
  local model=$2
  local out_dir=$3
  shift 3
  if [[ ! -d "$out_dir" ]]; then
    echo "  SKIP $out_dir: dir not found"
    return
  fi
  mkdir -p logs
  local logname=$(basename "$out_dir")
  nohup python3 scripts/train.py \
    --model "$model" --data-root "$DATA_ROOT" --resume latest --output-dir "$out_dir" \
    --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 \
    --image-size 128 --gpu "$gpu" --num-workers 8 --seed 1234 \
    "$@" \
    > "logs/${logname}.log" 2>&1 &
  echo "  $logname on GPU $gpu (PID $!)"
}

echo "Continuing LMB reg ablations + SIM_VQ + LMB main (all --resume latest)..."
echo "=========================================================================="

# GPU 0: LMB reg ablations (2)
run_reg 0 lmb_ablation_reg_none      0.0   0.0
run_reg 0 lmb_ablation_reg_peak_only 0.01  0.0

# GPU 1: SIM_VQ (reg_weak stopped, cap at 25)
run_main 1 sim_vq results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32 \
  --codebook-size 16384 --commitment-weight 10.0

# GPU 2: LMB reg ablation + LMB main (2)
run_reg 2 lmb_ablation_reg_strong    0.01  0.01
run_main 2 lmb results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd \
  --num-bins 16384 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 \
  --lambda-floor 0.0 --p-min 0.0 --tau-start 1.0 --tau-end 0.1

# GPU 3: (reg_bins_only stopped, cap at 25)

echo ""
echo "All 5 experiments launched (reg_bins_only, reg_weak stopped; epoch cap 25). Logs: logs/<run_name>.log"
disown

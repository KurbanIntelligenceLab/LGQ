#!/bin/bash
# Run the 5 LMB regularization ablations (none, weak, strong, bins_only, peak_only)
# distributed across GPUs 0–3. Keeps only best_model.pt and latest_model.pt (no
# per-epoch checkpoint_epoch_*.pt) to save space. Resumes from latest if present.
#
# Usage: ./shell/run_lmb_reg_ablations_save_space.sh
# Optional: GPU_IDS="0 1 2 3" to override GPUs (space-separated; 5th job uses first again).

set -e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
EPOCHS="${EPOCHS:-100}"

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
    --batch-size 32 --epochs "$EPOCHS" --lr 0.0003 --dim 128 --embedding-dim 128 \
    --image-size 128 --gpu "$gpu" --num-workers 8 --seed 1234 --run-name "$run_name" \
    --keep-only-best-latest \
    >> "$out_dir/train.log" 2>&1 &
  echo "  $run_name on GPU $gpu (PID $!) -> $out_dir/train.log"
}

echo "LMB regularization ablations (keep only best + latest checkpoints)"
echo "GPUs: ${GPU_IDS[*]}  Epochs: $EPOCHS"
echo "=========================================================================="

run_reg "${GPU_IDS[0]}" lmb_ablation_reg_none    0.0   0.0
run_reg "${GPU_IDS[1]:-${GPU_IDS[0]}}" lmb_ablation_reg_weak    0.002 0.002
run_reg "${GPU_IDS[2]:-${GPU_IDS[0]}}" lmb_ablation_reg_strong  0.01  0.01
run_reg "${GPU_IDS[3]:-${GPU_IDS[0]}}" lmb_ablation_reg_bins_only 0.0  0.01
run_reg "${GPU_IDS[0]}" lmb_ablation_reg_peak_only 0.01  0.0

echo ""
echo "All 5 LMB reg ablations launched. Only best_model.pt and latest_model.pt will be kept."
disown

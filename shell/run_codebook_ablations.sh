#!/bin/bash
# Run codebook ablations: FSQ, LFQ, VQ, SimVQ, LMB on 4K, 8K, 32K, 65K (exclude 16K - main runs that)
# Resumes from latest if checkpoint exists. Distributes across GPUs 0-3.

set -e
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
mkdir -p logs 2>/dev/null || true

run_ablation() {
  local gpu=$1
  shift 2  # drop gpu and out_log
  nohup python3 scripts/train.py --data-root "$DATA_ROOT" --gpu "$gpu" --keep-only-best-latest "$@" > /dev/null 2>&1 &
  echo "  PID: $! (GPU $gpu)"
}

echo "Launching codebook ablations (--resume latest where exists)..."
echo "=========================================================================="

# FSQ: 4K, 8K, 32K, 65K
run_ablation 0 results/fsq/fsq_cb4k/train.log --model fsq --resume latest --output-dir results/fsq/fsq_cb4k --levels 8 8 8 8 --run-name fsq_cb4k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 1 results/fsq/fsq_cb8k/train.log --model fsq --resume latest --output-dir results/fsq/fsq_cb8k --levels 16 8 8 8 --run-name fsq_cb8k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 2 results/fsq/fsq_cb32k/train.log --model fsq --resume latest --output-dir results/fsq/fsq_cb32k --levels 32 16 8 8 --run-name fsq_cb32k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 3 results/fsq/fsq_cb65k/train.log --model fsq --resume latest --output-dir results/fsq/fsq_cb65k --levels 64 16 8 8 --run-name fsq_cb65k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234

sleep 2

# LFQ: 4K, 8K, 32K, 65K
run_ablation 0 results/lfq/lfq_cb4k/train.log --model lfq --resume latest --output-dir results/lfq/lfq_cb4k --codebook-size 4096 --run-name lfq_cb4k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 1 results/lfq/lfq_cb8k/train.log --model lfq --resume latest --output-dir results/lfq/lfq_cb8k --codebook-size 8192 --run-name lfq_cb8k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 2 results/lfq/lfq_cb32k/train.log --model lfq --resume latest --output-dir results/lfq/lfq_cb32k --codebook-size 32768 --run-name lfq_cb32k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 3 results/lfq/lfq_cb65k/train.log --model lfq --resume latest --output-dir results/lfq/lfq_cb65k --codebook-size 65536 --run-name lfq_cb65k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234

sleep 2

# VQ: 4K, 8K, 32K, 65K
run_ablation 0 results/vq/vq_cb4k/train.log --model vq --resume latest --output-dir results/vq/vq_cb4k --codebook-size 4096 --run-name vq_cb4k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 1 results/vq/vq_cb8k/train.log --model vq --resume latest --output-dir results/vq/vq_cb8k --codebook-size 8192 --run-name vq_cb8k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 2 results/vq/vq_cb32k/train.log --model vq --resume latest --output-dir results/vq/vq_cb32k --codebook-size 32768 --run-name vq_cb32k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 3 results/vq/vq_cb65k/train.log --model vq --resume latest --output-dir results/vq/vq_cb65k --codebook-size 65536 --run-name vq_cb65k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234

sleep 2

# SimVQ: 4K, 8K, 32K, 65K
run_ablation 0 results/sim_vq/sim_vq_cb4k/train.log --model sim_vq --resume latest --output-dir results/sim_vq/sim_vq_cb4k --codebook-size 4096 --run-name sim_vq_cb4k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 1 results/sim_vq/sim_vq_cb8k/train.log --model sim_vq --resume latest --output-dir results/sim_vq/sim_vq_cb8k --codebook-size 8192 --run-name sim_vq_cb8k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 2 results/sim_vq/sim_vq_cb32k/train.log --model sim_vq --resume latest --output-dir results/sim_vq/sim_vq_cb32k --codebook-size 32768 --run-name sim_vq_cb32k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 3 results/sim_vq/sim_vq_cb65k/train.log --model sim_vq --resume latest --output-dir results/sim_vq/sim_vq_cb65k --codebook-size 65536 --run-name sim_vq_cb65k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234

sleep 2

# LMB: 4K, 8K, 32K, 65K
run_ablation 0 results/lmb/lmb_ablation_cb4k/train.log --model lmb --resume latest --output-dir results/lmb/lmb_ablation_cb4k --num-bins 4096 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 --tau-start 1.0 --tau-end 0.1 --run-name lmb_ablation_cb4k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 1 results/lmb/lmb_ablation_cb8k/train.log --model lmb --resume latest --output-dir results/lmb/lmb_ablation_cb8k --num-bins 8192 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 --tau-start 1.0 --tau-end 0.1 --run-name lmb_ablation_cb8k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 2 results/lmb/lmb_ablation_cb32k/train.log --model lmb --resume latest --output-dir results/lmb/lmb_ablation_cb32k --num-bins 32768 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 --tau-start 1.0 --tau-end 0.1 --run-name lmb_ablation_cb32k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234
run_ablation 3 results/lmb/lmb_ablation_cb65k/train.log --model lmb --resume latest --output-dir results/lmb/lmb_ablation_cb65k --num-bins 65536 --flatten-channels --lambda-peak 0.005 --lambda-bins 0.005 --tau-start 1.0 --tau-end 0.1 --run-name lmb_ablation_cb65k --batch-size 32 --epochs 100 --lr 0.0003 --dim 128 --embedding-dim 128 --image-size 128 --num-workers 8 --seed 1234

echo ""
echo "All 20 codebook ablations launched!"
echo "Logs disabled (output to /dev/null)"
disown

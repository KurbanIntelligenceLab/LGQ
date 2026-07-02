#!/bin/bash
# Quick smoke test: runs all 5 model types for 20 steps to verify the environment
# and DataLoader forkserver fix work correctly.
#
#SBATCH --job-name=smoke_test
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/smoke_test_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/smoke_test_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
STEPS=20
SEED=1234

echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(python3 --version)"
echo ""

run_test() {
    local model="$1"
    shift
    echo "========== Smoke test: $model =========="
    python3 scripts/train.py \
        --model "$model" \
        --data-root "$DATA_ROOT" \
        --image-size 256 \
        --batch-size 8 \
        --epochs 1 \
        --lr 3e-4 \
        --dim 256 \
        --embedding-dim 64 \
        --num-workers 0 \
        --seed "$SEED" \
        --run-name "smoke_${model}" \
        --max-steps "$STEPS" \
        --no-eval \
        --print-every 5 \
        "$@"
    echo "  PASSED: $model"
    echo ""
}

run_test vq  --codebook-size 4096 --commitment-weight 1.0 --decay 0.8
run_test fsq --levels 8 8 8 4 4
run_test lfq --codebook-size 4096
run_test sim_vq --codebook-size 4096
run_test lmb --num-bins 4096 --lambda-peak 0.005 --lambda-bins 0.005 \
             --tau-start 1.0 --tau-end 0.1 --flatten-channels --init-method random

echo "========== All smoke tests PASSED =========="

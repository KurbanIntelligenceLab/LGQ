#!/bin/bash
# Phase 0 quick test: M'=1000 on GPU
#
#SBATCH --job-name=phase0_test
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-00:15:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/phase0_test_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/phase0_test_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(python3 --version)"
echo ""

python3 phase0_datacentric.py \
    --checkpoint results/lmb/lmb_ablation_cb8k/checkpoints/best_model.pt \
    --data-root /N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet \
    --dataset imagenet100 \
    --num-samples 1000 \
    --batch-size 32 \
    --image-size 128 \
    --output-dir phase0_results \
    --num-workers 4 \
    --seed 42

echo ""
echo "Test complete. Results:"
cat phase0_results/summary.json

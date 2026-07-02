#!/bin/bash
# Phase 0: Data-Centric Codebook Initialisation — Full ImageNet, 256x256 encoder
# Uses a checkpoint from the 256-res LMB runs (embedding_dim=64)
# More samples & memory than the ImageNet-100 version since full dataset is 10x larger
#
#SBATCH --job-name=phase0_full
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-03:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/phase0_full_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/phase0_full_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"

# Use the first 256-res checkpoint that becomes available
# Default: 8K codebook run (middle size, likely representative)
CHECKPOINT="${1:-results/lmb/lmb_cb8192_256_s1234/checkpoints/best_model.pt}"
NUM_SAMPLES="${2:-200000}"
OUTPUT_DIR="${3:-phase0_results_full}"

echo "=============================================="
echo "Phase 0: Full ImageNet, 256x256 encoder"
echo "=============================================="
echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(python3 --version)"
echo "Checkpoint: $CHECKPOINT"
echo "Num samples: $NUM_SAMPLES"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Verify checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "The 256-res LMB runs may not have finished epoch 1 yet."
    echo "Check: ls results/lmb/lmb_cb*_256_s1234/checkpoints/"
    exit 1
fi

mkdir -p logs/slurm

python3 phase0_datacentric.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --dataset imagenet \
    --num-samples "$NUM_SAMPLES" \
    --batch-size 64 \
    --image-size 256 \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 8 \
    --seed 42

echo ""
echo "Phase 0 (Full ImageNet) complete. Results in $OUTPUT_DIR/"
echo "Summary:"
cat "$OUTPUT_DIR/summary.json"
echo ""
echo "Compare with ImageNet-100 results:"
echo "  cat phase0_results/summary.json"

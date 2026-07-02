#!/bin/bash
# Phase 0: Data-Centric Codebook Initialisation
# Single GPU, ~16GB RAM, under 1 hour
#
#SBATCH --job-name=phase0_datacentric
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-01:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/phase0_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/phase0_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
CHECKPOINT="${1:-results/lmb/lmb_cb8192_256_s1234/checkpoints/best_model.pt}"
NUM_SAMPLES="${2:-50000}"
DATASET="${3:-imagenet}"
IMAGE_SIZE="${4:-256}"
OUTPUT_DIR="${5:-phase0_results_256}"

echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(python3 --version)"
echo "Checkpoint: $CHECKPOINT"
echo "Num samples: $NUM_SAMPLES"
echo "Dataset: $DATASET"
echo "Image size: $IMAGE_SIZE"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Create logs directory
mkdir -p logs/slurm

python3 phase0_datacentric.py \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --dataset "$DATASET" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size 16 \
    --image-size "$IMAGE_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    --num-workers 0 \
    --seed 42

echo ""
echo "Phase 0 complete. Results in $OUTPUT_DIR/"
echo "Summary:"
cat "$OUTPUT_DIR/summary.json"

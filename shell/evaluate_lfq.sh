#!/bin/bash
# Evaluate Lookup Free Quantization (LFQ) model on ImageNet

set -e

# Change to project root directory
cd "$(dirname "$0")/.."  || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

# Default values
CHECKPOINT="${CHECKPOINT:-results/lfq/checkpoints/best_model.pt}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
OUTPUT_DIR="${OUTPUT_DIR:-results/lfq}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
GPU="${GPU:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        
            COMPUTE_FID="
            shift
            ;;
        
            FID_SAMPLES="$2"
            shift 2
            ;;
        --save-samples)
            SAVE_SAMPLES="--save-samples"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Evaluating LFQ model on ImageNet"
echo "================================"
echo "Checkpoint: $CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "GPU: $GPU"
echo ""

python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
     \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    --batch-size "$BATCH_SIZE" \
    --gpu "$GPU" \
    --num-workers "$NUM_WORKERS" \
    ${NUM_SAMPLES:+--num-samples "$NUM_SAMPLES"} \
     \
    ${FID_SAMPLES:+
    ${SAVE_SAMPLES:-}

echo ""
echo "Evaluation completed!"


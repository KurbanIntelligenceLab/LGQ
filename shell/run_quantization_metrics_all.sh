#!/bin/bash
# Run quantization metrics analysis on all main models

set -e

# Main model checkpoints (16K codebook models)
CHECKPOINTS=(
    "results/fsq/fsq_lv16-16-8-8_bs32_lr3e-4_dim128_20260123_184222_deab/checkpoints/latest_model.pt"
    "results/vq/vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0/checkpoints/best_model.pt"
    "results/lfq/lfq_cb16384_bs32_lr3e-4_dim128_20260123_221044_6f79/checkpoints/latest_model.pt"
    "results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32/checkpoints/latest_model.pt"
    "results/lmb/lmb_nb16384_tau1.0-0.1_bs32_lr3e-4_dim128_20260122_002044_cefd/checkpoints/latest_model.pt"
)

# GPUs to use (prioritize emptier ones)
GPUS=(1 0 2 3)

OUTPUT_DIR="results/plots"
NUM_IMAGES=1000

cd /workspace/vector-quantize

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to run analysis on a checkpoint
run_analysis() {
    local checkpoint=$1
    local gpu=$2
    local model_name=$(basename $(dirname $(dirname $checkpoint)))
    
    echo "=========================================="
    echo "Running analysis for: $model_name"
    echo "Checkpoint: $checkpoint"
    echo "GPU: $gpu"
    echo "=========================================="
    
    CUDA_VISIBLE_DEVICES=$gpu python3 scripts/analyze_quantization_metrics.py \
        --checkpoint "$checkpoint" \
        --num-images $NUM_IMAGES \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 32 \
        2>&1 | tee "$OUTPUT_DIR/${model_name}_metrics.log"
    
    echo "Completed: $model_name"
    echo ""
}

# Run analyses in parallel (one per GPU)
for i in "${!CHECKPOINTS[@]}"; do
    checkpoint="${CHECKPOINTS[$i]}"
    gpu="${GPUS[$i % ${#GPUS[@]}]}"
    
    # Check if checkpoint exists
    if [ ! -f "$checkpoint" ]; then
        echo "Warning: Checkpoint not found: $checkpoint"
        continue
    fi
    
    # Run in background
    run_analysis "$checkpoint" "$gpu" &
done

# Wait for all background jobs
wait

echo "=========================================="
echo "All analyses completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

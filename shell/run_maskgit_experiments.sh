#!/bin/bash
# Run MaskGIT transformer training with different VQ models
#
# Usage:
#   ./shell/run_maskgit_experiments.sh vq        # Run VQ only
#   ./shell/run_maskgit_experiments.sh sim_vq    # Run SimVQ only
#   ./shell/run_maskgit_experiments.sh both      # Run both sequentially
#
# Each experiment trains a MaskGIT transformer using the frozen VQ tokenizer
# to learn masked image generation on 128x128 images with 64 tokens.

set -e

cd "$(dirname "$0")/.."

# Default settings (smaller for testing)
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
ACCUM_GRAD=${ACCUM_GRAD:-4}
LR=${LR:-1e-4}
N_LAYERS=${N_LAYERS:-12}
DIM=${DIM:-512}
HIDDEN_DIM=${HIDDEN_DIM:-2048}
GPU=${GPU:-0}

# Data path - update if needed
DATA_PATH=${DATA_PATH:-"data/imagenet"}
# Resume: set to "latest" or "best" to resume from checkpoint (e.g. RESUME=latest)
RESUME=${RESUME:-""}

run_experiment() {
    local QUANTIZER=$1
    echo "============================================================"
    echo "Running MaskGIT with ${QUANTIZER} tokenizer"
    if [ -n "$RESUME" ]; then echo "Resume: $RESUME"; fi
    echo "============================================================"
    
    RESUME_ARGS=""
    if [ -n "$RESUME" ]; then
        RESUME_ARGS="--resume $RESUME"
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU python3 MaskGIT-pytorch-main/train_maskgit_custom.py \
        --quantizer $QUANTIZER \
        --dataset-path $DATA_PATH \
        --image-size 128 \
        --batch-size $BATCH_SIZE \
        --accum-grad $ACCUM_GRAD \
        --epochs $EPOCHS \
        --learning-rate $LR \
        --n-layers $N_LAYERS \
        --dim $DIM \
        --hidden-dim $HIDDEN_DIM \
        --sample-every 5 \
        --save-every 10 \
        --num-workers 4 \
        $RESUME_ARGS
}

case "${1:-both}" in
    vq)
        run_experiment vq
        ;;
    sim_vq)
        run_experiment sim_vq
        ;;
    fsq)
        run_experiment fsq
        ;;
    lfq)
        run_experiment lfq
        ;;
    lmb)
        run_experiment lmb
        ;;
    lmb_fixed)
        run_experiment lmb_fixed
        ;;
    both)
        echo "Running VQ and SimVQ experiments sequentially..."
        run_experiment vq
        run_experiment sim_vq
        ;;
    main)
        echo "Running all main-experiment MaskGIT runs (FSQ, LFQ, LMB, LMB-Fixed, SimVQ, VQ)..."
        for q in fsq lfq lmb lmb_fixed sim_vq vq; do
            run_experiment $q
        done
        ;;
    all)
        echo "Running all quantizer experiments..."
        for q in vq sim_vq fsq lfq lmb lmb_fixed; do
            run_experiment $q
        done
        ;;
    *)
        echo "Unknown quantizer: $1"
        echo "Usage: $0 [vq|sim_vq|fsq|lfq|lmb|lmb_fixed|both|main|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Training complete!"
echo "Results saved to: results/maskgit/"
echo "============================================================"

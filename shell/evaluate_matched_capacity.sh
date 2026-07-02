#!/bin/bash
# Evaluate FSQ and SimVQ at matched effective capacity (LMB active codes K).
# 1. Run LMB eval → get K = active_codes
# 2. FSQ at K (prune indices to top-K)
# 3. SimVQ at K (mask codebook to top-K)
# Compare rFID / LPIPS at matched K.

set -e

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

REFERENCE="${REFERENCE:-results/lmb/checkpoints/best_model.pt}"
FSQ_CHECKPOINT="${FSQ_CHECKPOINT:-results/fsq/checkpoints/best_model.pt}"
SIMVQ_CHECKPOINT="${SIMVQ_CHECKPOINT:-results/sim_vq/checkpoints/best_model.pt}"
DATA_ROOT="${DATA_ROOT:-data/imagenet}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_SAMPLES="${NUM_SAMPLES:-}"
FID_SAMPLES="${FID_SAMPLES:-5000}"
GPU="${GPU:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --reference) REFERENCE="$2"; shift 2 ;;
        --fsq) FSQ_CHECKPOINT="$2"; shift 2 ;;
        --sim-vq) SIMVQ_CHECKPOINT="$2"; shift 2 ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --fid-samples) FID_SAMPLES="$2"; shift 2 ;;
        --capacity-k) CAPACITY_K="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --output) OUTPUT_JSON="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Matched effective capacity evaluation"
echo "======================================"
echo "Reference (LMB): $REFERENCE"
echo "FSQ checkpoint:  $FSQ_CHECKPOINT"
echo "SimVQ checkpoint: $SIMVQ_CHECKPOINT"
echo "Data root: $DATA_ROOT"
echo "FID samples: $FID_SAMPLES"
echo ""

ARGS=(
    --data-root "$DATA_ROOT"
    --batch-size "$BATCH_SIZE"
    --fid-samples "$FID_SAMPLES"
    --gpu "$GPU"
    --num-workers "$NUM_WORKERS"
)
[[ -n "$CAPACITY_K" ]] && ARGS+=(--capacity-k "$CAPACITY_K")
[[ -z "$CAPACITY_K" ]] && ARGS+=(--reference-checkpoint "$REFERENCE")
[[ -n "$NUM_SAMPLES" ]] && ARGS+=(--num-samples "$NUM_SAMPLES")
[[ -n "$OUTPUT_JSON" ]] && ARGS+=(--output "$OUTPUT_JSON")

python3 scripts/evaluate_matched_capacity.py \
    "${ARGS[@]}" \
    --fsq-checkpoint "$FSQ_CHECKPOINT" \
    --sim-vq-checkpoint "$SIMVQ_CHECKPOINT"

echo ""
echo "Done."

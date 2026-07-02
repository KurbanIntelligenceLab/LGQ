#!/bin/bash
# Submit LMB legacy-decoder experiments across all codebook sizes.
# Uses old decoder (ch = max(32, embedding_dim)) with perceptual+GAN losses.
#
# Usage:
#   bash shell/slurm/run_lmb_legacy_all.sh
#   bash shell/slurm/run_lmb_legacy_all.sh --dry-run

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/Data/imagenet}"
SEED="${SEED:-1234}"
DRY_RUN=0

CODEBOOK_SIZES="4096 8192 16384 32768 65536"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=1; shift ;;
        --sizes)     CODEBOOK_SIZES="$2"; shift 2 ;;
        --seed)      SEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$PROJECT_ROOT/logs/slurm"

TRAIN_SCRIPT="$SCRIPT_DIR/train_lmb_legacy.sh"
if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: $TRAIN_SCRIPT not found"
    exit 1
fi

echo "============================================================"
echo "  LMB legacy-decoder experiments (old decoder + perc/GAN)"
echo "  Script: $TRAIN_SCRIPT"
echo "  Sizes : $CODEBOOK_SIZES"
echo "  Seed  : $SEED"
echo "  Data  : $DATA_ROOT"
echo "  Dry   : $DRY_RUN"
echo "============================================================"

for CB in $CODEBOOK_SIZES; do
    TAG="lmb_legacy_cb${CB}_256_s${SEED}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] sbatch --job-name=$TAG --export=ALL,CODEBOOK_SIZE=$CB,DATA_ROOT=$DATA_ROOT,SEED=$SEED $TRAIN_SCRIPT"
    else
        JOB_ID=$(sbatch \
            --job-name="$TAG" \
            --export=ALL,CODEBOOK_SIZE="$CB",DATA_ROOT="$DATA_ROOT",SEED="$SEED" \
            "$TRAIN_SCRIPT" | awk '{print $NF}')
        echo "  Submitted $TAG -> job $JOB_ID"
    fi
done

echo ""
if [[ $DRY_RUN -eq 0 ]]; then
    echo "All jobs submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f ${PROJECT_ROOT}/logs/slurm/lmb_legacy_cb*_256_s${SEED}_*.out"
fi

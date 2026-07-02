#!/bin/bash
# Submit all 256x256 ImageNet experiments to SLURM.
#
# Experiment matrix (5 models x 5 codebook sizes = 25 jobs):
#   Models     : lmb, fsq, vq, lfq, sim_vq
#   Sizes      : 4096, 8192, 16384, 32768, 65536
#   Seed       : 1234 (default; pass --seeds "1234 5678 ..." for multi-seed later)
#
# Usage:
#   bash shell/slurm/submit_all.sh                             # submit all
#   bash shell/slurm/submit_all.sh --data-root /path/imagenet
#   bash shell/slurm/submit_all.sh --models "lmb fsq"         # subset of models
#   bash shell/slurm/submit_all.sh --sizes "16384"            # single size
#   bash shell/slurm/submit_all.sh --seeds "1234 42 99"       # multi-seed (for std)
#   bash shell/slurm/submit_all.sh --dry-run                  # print only

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Defaults
DRY_RUN=0
DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
MODELS="lmb fsq vq lfq sim_vq rot_vq softvq ibq bsq"
SIZES="4096 8192 16384 32768 65536"
SEEDS="1234"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)   DRY_RUN=1; shift ;;
        --data-root) DATA_ROOT="$2"; shift 2 ;;
        --models)    MODELS="$2"; shift 2 ;;
        --sizes)     SIZES="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$PROJECT_ROOT/logs/slurm"

N_MODELS=$(echo $MODELS | wc -w)
N_SIZES=$(echo $SIZES | wc -w)
N_SEEDS=$(echo $SEEDS | wc -w)
TOTAL=$(( N_MODELS * N_SIZES * N_SEEDS ))

echo "============================================================"
echo "  Submitting 256x256 ImageNet experiments"
echo "  Project : $PROJECT_ROOT"
echo "  Data    : $DATA_ROOT"
echo "  Models  : $MODELS"
echo "  Sizes   : $SIZES"
echo "  Seeds   : $SEEDS"
echo "  Jobs    : $TOTAL"
echo "  Dry run : $DRY_RUN"
echo "============================================================"

submit() {
    local script="$1"
    local cb="$2"
    local seed="$3"
    local model
    model=$(basename "$script" .sh | sed 's/^train_//')
    local tag="${model}_cb${cb}_256_s${seed}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY-RUN] sbatch --job-name=$tag --export=ALL,CODEBOOK_SIZE=$cb,DATA_ROOT=$DATA_ROOT,SEED=$seed $script"
    else
        local job_id
        job_id=$(sbatch \
            --job-name="$tag" \
            --export=ALL,CODEBOOK_SIZE="$cb",DATA_ROOT="$DATA_ROOT",SEED="$seed" \
            "$script" | awk '{print $NF}')
        echo "  Submitted $tag -> job $job_id"
    fi
}

for MODEL in $MODELS; do
    SCRIPT="$SCRIPT_DIR/train_${MODEL}.sh"
    if [[ ! -f "$SCRIPT" ]]; then
        echo "WARNING: script not found: $SCRIPT — skipping"
        continue
    fi
    echo ""
    echo "--- $MODEL ---"
    if [[ "$MODEL" == "bsq" ]]; then
        # BSQ has implicit codebook (2^dim), no codebook-size sweep
        for SEED in $SEEDS; do
            submit "$SCRIPT" "0" "$SEED"
        done
    else
        for CB in $SIZES; do
            for SEED in $SEEDS; do
                submit "$SCRIPT" "$CB" "$SEED"
            done
        done
    fi
done

echo ""
if [[ $DRY_RUN -eq 0 ]]; then
    echo "All $TOTAL jobs submitted. Monitor with:"
    echo "  squeue -u \$USER"
    echo "  tail -f $PROJECT_ROOT/logs/slurm/<tag>_<jobid>.out"
fi

#!/bin/bash
# Resume LMB 256-res training with SLURM dependency chaining
# Each codebook size gets 5 chained 2-day jobs (10 days total, enough for 100 epochs)
#
# Usage: bash resume_lmb_256.sh

cd /N/project/de_briujn_graph/Projects/vector-quantize
mkdir -p logs/slurm

PREAMBLE="module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1"

COMMON_ARGS="--data-root /N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet \
    --dataset imagenet \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --epochs 100 \
    --lr 3e-4 \
    --seed 1234 \
    --dim 256 \
    --embedding-dim 64 \
    --image-size 256 \
    --compression-factor 16 \
    --num-residual-layers 2 \
    --keep-only-best-latest \
    --fid-samples 10000 \
    --csv-log-every 50 \
    --print-every 100 \
    --amp \
    --num-workers 0 \
    --resume latest"

LMB_ARGS="--flatten-channels \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1"

SBATCH_COMMON="--partition=gpu -A r00432 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2-00:00:00"

CHAIN_LENGTH=5  # 5 x 2 days = 10 days max

for CB in 8192 16384 32768 65536; do
    RUN_NAME="lmb_cb${CB}_256_s1234"
    OUTPUT_DIR="results/lmb/${RUN_NAME}"

    # Verify checkpoint exists
    if [ ! -f "${OUTPUT_DIR}/checkpoints/latest_model.pt" ]; then
        echo "SKIP cb${CB}: no checkpoint at ${OUTPUT_DIR}/checkpoints/latest_model.pt"
        continue
    fi

    echo "Submitting chain of ${CHAIN_LENGTH} jobs for ${RUN_NAME}..."

    PREV_JOB=""
    for i in $(seq 1 $CHAIN_LENGTH); do
        DEP_FLAG=""
        if [ -n "$PREV_JOB" ]; then
            DEP_FLAG="--dependency=afterany:${PREV_JOB}"
        fi

        JOB_ID=$(sbatch $SBATCH_COMMON $DEP_FLAG \
            --job-name="${RUN_NAME}" \
            --output="logs/slurm/${RUN_NAME}_%j.out" \
            --error="logs/slurm/${RUN_NAME}_%j.err" \
            --wrap="
$PREAMBLE
python3 scripts/train.py --model lmb $COMMON_ARGS $LMB_ARGS \
    --num-bins $CB \
    --run-name $RUN_NAME \
    --output-dir $OUTPUT_DIR
" | awk '{print $4}')

        echo "  Chain $i/$CHAIN_LENGTH: job $JOB_ID ${DEP_FLAG:+(depends on $PREV_JOB)}"
        PREV_JOB=$JOB_ID
    done
    echo ""
done

echo "All chains submitted. Monitor with: squeue -u \$USER"

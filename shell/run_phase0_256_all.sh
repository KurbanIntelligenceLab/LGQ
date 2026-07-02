#!/bin/bash
# Phase 0: Data-Centric analysis for all LMB 256-res codebook sizes
# Uses epoch-1 checkpoints. Smaller batch size to avoid OOM on 256-res.
#
# Usage: bash run_phase0_256_all.sh

cd /N/project/de_briujn_graph/Projects/vector-quantize
mkdir -p logs/slurm

PREAMBLE="module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1"

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"

for CB in 8192 16384 32768 65536; do
    CKPT="results/lmb/lmb_cb${CB}_256_s1234/checkpoints/best_model.pt"
    OUTDIR="phase0_results_256_cb${CB}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP cb${CB}: no checkpoint at $CKPT"
        continue
    fi

    echo "Submitting phase0 for LMB cb${CB} (256-res)..."
    sbatch --partition=gpu -A r00432 \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=64G \
        --time=0-03:00:00 \
        --job-name="p0_256_cb${CB}" \
        --output="logs/slurm/phase0_256_cb${CB}_%j.out" \
        --error="logs/slurm/phase0_256_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE

python3 phase0_datacentric.py \
    --checkpoint $CKPT \
    --data-root $DATA_ROOT \
    --dataset imagenet \
    --num-samples 50000 \
    --batch-size 4 \
    --image-size 256 \
    --output-dir $OUTDIR \
    --num-workers 0 \
    --seed 42
"
done

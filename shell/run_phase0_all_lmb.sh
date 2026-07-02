#!/bin/bash
# Phase 0: Data-Centric analysis for all LMB 256-res codebook sizes
# Submits one job per codebook size
#
# Usage: bash run_phase0_all_lmb.sh

cd /N/project/de_briujn_graph/Projects/vector-quantize

for CB in 4096 8192 16384 32768 65536; do
    CKPT="results/lmb/lmb_cb${CB}_256_s1234/checkpoints/best_model.pt"
    OUTDIR="phase0_results_256_cb${CB}"

    if [ ! -f "$CKPT" ]; then
        echo "SKIP cb${CB}: no checkpoint at $CKPT"
        continue
    fi

    echo "Submitting phase0 for LMB cb${CB}..."
    sbatch --job-name="p0_cb${CB}" \
        --partition=gpu \
        -A r00432 \
        --gres=gpu:1 \
        --cpus-per-task=8 \
        --mem=32G \
        --time=0-01:00:00 \
        --output="logs/slurm/phase0_cb${CB}_%j.out" \
        --error="logs/slurm/phase0_cb${CB}_%j.err" \
        --wrap="
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1

python3 phase0_datacentric.py \
    --checkpoint $CKPT \
    --data-root /N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet \
    --dataset imagenet \
    --num-samples 50000 \
    --batch-size 16 \
    --image-size 256 \
    --output-dir $OUTDIR \
    --num-workers 0 \
    --seed 42
"
done

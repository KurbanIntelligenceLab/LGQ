#!/bin/bash
# Phase 0 (256-dim) + LMB codebook sweep with FIXED decoder (ch = base_ch * 2)
#
# 1. Run Phase 0 for each codebook size using existing LMB 256 checkpoints (encoder only)
# 2. Train LMB from scratch with fixed decoder + Phase 0 codebook init
#
# Usage: bash run_lmb256_phase0_sweep.sh

set -e
cd /N/project/de_briujn_graph/Projects/vector-quantize
mkdir -p logs/slurm

PREAMBLE="module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1"

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
SBATCH_COMMON="--partition=gpu -A r00432 --gres=gpu:1 --cpus-per-task=8 --mem=64G"

# Use the LMB 8192 checkpoint for Phase 0 (encoder is fine, decoder loaded with strict=False)
PHASE0_CKPT="results/lmb/lmb_cb8192_256_s1234/checkpoints/best_model.pt"

# ---------- STEP 1: Phase 0 for each codebook size ----------
PHASE0_JOBS=""
for CB in 4096 8192 16384 32768 65536; do
    P0_OUTDIR="phase0_results_256_cb${CB}"

    echo "Submitting Phase 0 for cb${CB} (256-dim)..."
    P0_JOB=$(sbatch --parsable $SBATCH_COMMON \
        --time=0-04:00:00 \
        --job-name="p0_256_${CB}" \
        --output="logs/slurm/p0_256_${CB}_%j.out" \
        --error="logs/slurm/p0_256_${CB}_%j.err" \
        --wrap="
$PREAMBLE

python3 phase0_datacentric.py \
    --checkpoint $PHASE0_CKPT \
    --data-root $DATA_ROOT \
    --dataset imagenet \
    --num-samples 50000 \
    --batch-size 4 \
    --image-size 256 \
    --output-dir $P0_OUTDIR \
    --override-K $CB \
    --num-workers 0 \
    --seed 42
")
    echo "  Phase 0 cb${CB} submitted as job $P0_JOB"
    PHASE0_JOBS="${PHASE0_JOBS}:${P0_JOB}"
done

echo ""
echo "Phase 0 jobs: ${PHASE0_JOBS}"
echo ""

# ---------- STEP 2: LMB training (depends on Phase 0) ----------
for CB in 4096 8192 16384 32768 65536; do
    P0_OUTDIR="phase0_results_256_cb${CB}"
    CODEBOOK_NPY="${P0_OUTDIR}/codebook_datacentric.npy"
    TAG="lmb_cb${CB}_256_decfix_s1234"

    echo "Submitting LMB training cb${CB} (depends on Phase 0)..."
    sbatch $SBATCH_COMMON \
        --time=2-00:00:00 \
        --dependency=afterok${PHASE0_JOBS} \
        --job-name="lmb_${CB}_df" \
        --output="logs/slurm/lmb_${CB}_decfix_%j.out" \
        --error="logs/slurm/lmb_${CB}_decfix_%j.err" \
        --wrap="
$PREAMBLE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/train.py \
    --model lmb \
    --data-root $DATA_ROOT \
    --dataset imagenet \
    --image-size 256 \
    --batch-size 4 \
    --grad-accum-steps 8 \
    --amp \
    --epochs 100 \
    --lr 3e-4 \
    --dim 256 \
    --embedding-dim 64 \
    --num-bins $CB \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --tau-start 1.0 \
    --tau-end 0.1 \
    --flatten-channels \
    --codebook-init $CODEBOOK_NPY \
    --num-workers 0 \
    --seed 1234 \
    --run-name $TAG \
    --keep-only-best-latest \
    --fid-samples 10000 \
    --csv-log-every 50 \
    --print-every 100 \
    --gpu 0
"
done

echo ""
echo "All jobs submitted. LMB training will start after Phase 0 completes."
echo "Monitor with: squeue -u \$USER"

#!/bin/bash
# Re-queue all non-lmb-256 experiments with dependency on lmb 256 jobs.
# These will start only after all 5 lmb_cb*_256 jobs complete.
#
# Auto-generated on 2026-04-02 to restore cancelled jobs.

set -e
cd /N/project/de_briujn_graph/Projects/vector-quantize
mkdir -p logs/slurm

# Dependency: wait for all 5 lmb 256 jobs
DEP="--dependency=afterany:6727312:6727313:6727314:6727315:6727316"

SBATCH_COMMON="--partition=gpu -A r00432 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2-00:00:00"

PREAMBLE="module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1"

COMMON_ARGS="--data-root /N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet \
    --dataset imagenet100 \
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
    --num-workers 0"

COUNT=0

echo "============================================================"
echo "  Re-queuing experiments (deferred until lmb 256 jobs finish)"
echo "============================================================"

# =============================================
# 1) ImageNet-100 experiments (6 models x 5 CB sizes = 30 jobs)
# =============================================
for CB in 4096 8192 16384 32768 65536; do

    # --- VQ ---
    sbatch $SBATCH_COMMON $DEP \
        --job-name="vq_100_${CB}" \
        --output="logs/slurm/vq_100_cb${CB}_%j.out" \
        --error="logs/slurm/vq_100_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model vq $COMMON_ARGS \
    --codebook-size $CB \
    --commitment-weight 1.0 \
    --decay 0.8
"
    COUNT=$((COUNT + 1))

    # --- RotVQ ---
    sbatch $SBATCH_COMMON $DEP \
        --job-name="rotvq_100_${CB}" \
        --output="logs/slurm/rotvq_100_cb${CB}_%j.out" \
        --error="logs/slurm/rotvq_100_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model rot_vq $COMMON_ARGS \
    --codebook-size $CB \
    --commitment-weight 1.0 \
    --decay 0.8
"
    COUNT=$((COUNT + 1))

    # --- FSQ ---
    case $CB in
        4096)  LEVELS="8 8 8 8" ;;
        8192)  LEVELS="8 8 8 8 2" ;;
        16384) LEVELS="8 8 8 8 4" ;;
        32768) LEVELS="8 8 8 8 8" ;;
        65536) LEVELS="8 8 8 8 16" ;;
    esac
    sbatch $SBATCH_COMMON $DEP \
        --job-name="fsq_100_${CB}" \
        --output="logs/slurm/fsq_100_cb${CB}_%j.out" \
        --error="logs/slurm/fsq_100_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model fsq $COMMON_ARGS \
    --levels $LEVELS
"
    COUNT=$((COUNT + 1))

    # --- LFQ ---
    sbatch $SBATCH_COMMON $DEP \
        --job-name="lfq_100_${CB}" \
        --output="logs/slurm/lfq_100_cb${CB}_%j.out" \
        --error="logs/slurm/lfq_100_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model lfq $COMMON_ARGS \
    --codebook-size $CB \
    --entropy-loss-weight 0.1 \
    --diversity-gamma 1.0 \
    --spherical
"
    COUNT=$((COUNT + 1))

    # --- SimVQ ---
    sbatch $SBATCH_COMMON $DEP \
        --job-name="simvq_100_${CB}" \
        --output="logs/slurm/simvq_100_cb${CB}_%j.out" \
        --error="logs/slurm/simvq_100_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model sim_vq $COMMON_ARGS \
    --codebook-size $CB \
    --use-mlp \
    --commitment-weight 10.0
"
    COUNT=$((COUNT + 1))

    # --- LMB ---
    sbatch $SBATCH_COMMON $DEP \
        --job-name="lmb_100_${CB}" \
        --output="logs/slurm/lmb_100_cb${CB}_%j.out" \
        --error="logs/slurm/lmb_100_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model lmb $COMMON_ARGS \
    --num-bins $CB \
    --flatten-channels \
    --lambda-peak 0.005 \
    --lambda-bins 0.005 \
    --lambda-floor 0.0 \
    --p-min 0.0 \
    --tau-start 1.0 \
    --tau-end 0.1
"
    COUNT=$((COUNT + 1))

done

# =============================================
# 2) 256-res non-LMB (full ImageNet, cb8192) — 4 jobs
# =============================================
COMMON_ARGS_256="--data-root /N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet \
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
    --num-workers 0"

# VQ 256 8k
sbatch $SBATCH_COMMON $DEP \
    --job-name="vq_256_8k" \
    --output="logs/slurm/vq_256_8k_%j.out" \
    --error="logs/slurm/vq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model vq $COMMON_ARGS_256 \
    --codebook-size 8192 \
    --commitment-weight 1.0 \
    --decay 0.8
"
COUNT=$((COUNT + 1))

# FSQ 256 8k
sbatch $SBATCH_COMMON $DEP \
    --job-name="fsq_256_8k" \
    --output="logs/slurm/fsq_256_8k_%j.out" \
    --error="logs/slurm/fsq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model fsq $COMMON_ARGS_256 \
    --levels 8 8 8 8 2
"
COUNT=$((COUNT + 1))

# LFQ 256 8k
sbatch $SBATCH_COMMON $DEP \
    --job-name="lfq_256_8k" \
    --output="logs/slurm/lfq_256_8k_%j.out" \
    --error="logs/slurm/lfq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model lfq $COMMON_ARGS_256 \
    --codebook-size 8192 \
    --entropy-loss-weight 0.1 \
    --diversity-gamma 1.0 \
    --spherical
"
COUNT=$((COUNT + 1))

# SimVQ 256 8k
sbatch $SBATCH_COMMON $DEP \
    --job-name="simvq_256_8k" \
    --output="logs/slurm/simvq_256_8k_%j.out" \
    --error="logs/slurm/simvq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model sim_vq $COMMON_ARGS_256 \
    --codebook-size 8192 \
    --use-mlp \
    --commitment-weight 10.0
"
COUNT=$((COUNT + 1))

# =============================================
# 3) Special jobs
# =============================================

# smoke_lmb_decfix
sbatch --partition=gpu -A r00432 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=0-00:30:00 $DEP \
    --job-name="smoke_lmb_decfix" \
    --output="logs/slurm/smoke_lmb_decfix_%j.out" \
    --error="logs/slurm/smoke_lmb_decfix_%j.err" \
    /N/project/de_briujn_graph/Projects/vector-quantize/smoke_test_lmb_decfix.sh
COUNT=$((COUNT + 1))

# lgq-hier (external project)
sbatch $DEP \
    /N/project/de_briujn_graph/lgq_ts/run_hierarchical.sh
COUNT=$((COUNT + 1))

echo ""
echo "Re-queued $COUNT jobs with dependency on lmb 256 jobs (6727312-6727316)."
echo "These will start after all lmb 256 jobs complete."
echo "Monitor with: squeue -u \$USER"

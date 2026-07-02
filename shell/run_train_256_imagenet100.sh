#!/bin/bash
# Train ALL quantization models at 256-res across ALL codebook sizes on ImageNet-100
# Models: VQ, FSQ, LFQ, SimVQ, LMB, RotVQ
# Codebook sizes: 4096, 8192, 16384, 32768, 65536
# Total: 6 models x 5 sizes = 30 jobs
#
# Usage: bash run_train_256_imagenet100.sh

cd /N/project/de_briujn_graph/Projects/vector-quantize
mkdir -p logs/slurm

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

SBATCH_COMMON="--partition=gpu -A r00432 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2-00:00:00"

PREAMBLE="module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1"

COUNT=0

for CB in 4096 8192 16384 32768 65536; do

    # --- VQ ---
    echo "Submitting VQ cb${CB} (ImageNet-100)..."
    sbatch $SBATCH_COMMON \
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
    echo "Submitting RotVQ cb${CB} (ImageNet-100)..."
    sbatch $SBATCH_COMMON \
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
    # Map codebook size to FSQ levels
    case $CB in
        4096)  LEVELS="8 8 8 8" ;;
        8192)  LEVELS="8 8 8 8 2" ;;
        16384) LEVELS="8 8 8 8 4" ;;
        32768) LEVELS="8 8 8 8 8" ;;
        65536) LEVELS="8 8 8 8 16" ;;
    esac
    echo "Submitting FSQ cb${CB} levels=[${LEVELS}] (ImageNet-100)..."
    sbatch $SBATCH_COMMON \
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
    echo "Submitting LFQ cb${CB} (ImageNet-100)..."
    sbatch $SBATCH_COMMON \
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
    echo "Submitting SimVQ cb${CB} (ImageNet-100)..."
    sbatch $SBATCH_COMMON \
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
    echo "Submitting LMB cb${CB} (ImageNet-100)..."
    sbatch $SBATCH_COMMON \
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

    echo "---"
done

echo ""
echo "Submitted $COUNT jobs (6 models x 5 codebook sizes) on ImageNet-100."
echo "Monitor with: squeue -u \$USER"

#!/bin/bash
# Train ALL quantization models at 256-res across ALL codebook sizes on full ImageNet
# Models: VQ, RotVQ, FSQ, LFQ, SimVQ, LMB, SoftVQ, IBQ, BSQ
# Codebook sizes: 4096, 8192, 16384, 32768, 65536
# Total: 9 models x 5 sizes = 45 jobs
# Perceptual (LPIPS) loss enabled
#
# Usage: bash run_all_models_all_sizes.sh

cd /N/project/de_briujn_graph/Projects/vector-quantize
mkdir -p logs/slurm

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
    --perceptual-weight 0.1 \
    --resume latest \
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
    echo "Submitting VQ cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="vq_full_${CB}" \
        --output="logs/slurm/vq_full_cb${CB}_%j.out" \
        --error="logs/slurm/vq_full_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model vq $COMMON_ARGS \
    --codebook-size $CB \
    --commitment-weight 1.0 \
    --decay 0.8
"
    COUNT=$((COUNT + 1))

    # --- RotVQ ---
    echo "Submitting RotVQ cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="rotvq_full_${CB}" \
        --output="logs/slurm/rotvq_full_cb${CB}_%j.out" \
        --error="logs/slurm/rotvq_full_cb${CB}_%j.err" \
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
    echo "Submitting FSQ cb${CB} levels=[${LEVELS}] (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="fsq_full_${CB}" \
        --output="logs/slurm/fsq_full_cb${CB}_%j.out" \
        --error="logs/slurm/fsq_full_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model fsq $COMMON_ARGS \
    --levels $LEVELS
"
    COUNT=$((COUNT + 1))

    # --- LFQ ---
    echo "Submitting LFQ cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="lfq_full_${CB}" \
        --output="logs/slurm/lfq_full_cb${CB}_%j.out" \
        --error="logs/slurm/lfq_full_cb${CB}_%j.err" \
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
    echo "Submitting SimVQ cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="simvq_full_${CB}" \
        --output="logs/slurm/simvq_full_cb${CB}_%j.out" \
        --error="logs/slurm/simvq_full_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model sim_vq $COMMON_ARGS \
    --codebook-size $CB \
    --use-mlp \
    --commitment-weight 10.0
"
    COUNT=$((COUNT + 1))

    # --- LMB ---
    echo "Submitting LMB cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="lmb_full_${CB}" \
        --output="logs/slurm/lmb_full_cb${CB}_%j.out" \
        --error="logs/slurm/lmb_full_cb${CB}_%j.err" \
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

    # --- SoftVQ ---
    echo "Submitting SoftVQ cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="softvq_full_${CB}" \
        --output="logs/slurm/softvq_full_cb${CB}_%j.out" \
        --error="logs/slurm/softvq_full_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model softvq $COMMON_ARGS \
    --codebook-size $CB \
    --softvq-tau 0.07 \
    --softvq-entropy-ratio 0.01 \
    --softvq-l2-norm
"
    COUNT=$((COUNT + 1))

    # --- IBQ ---
    echo "Submitting IBQ cb${CB} (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="ibq_full_${CB}" \
        --output="logs/slurm/ibq_full_cb${CB}_%j.out" \
        --error="logs/slurm/ibq_full_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model ibq $COMMON_ARGS \
    --codebook-size $CB \
    --ibq-beta 0.25 \
    --ibq-entropy-weight 0.1 \
    --ibq-entropy-temp 0.01
"
    COUNT=$((COUNT + 1))

    # --- BSQ ---
    # BSQ uses binary quantization (implicit codebook = 2^embedding_dim).
    # Vary --bsq-group-size = log2(CB) so entropy sub-codebook matches.
    case $CB in
        4096)  GS=12 ;;
        8192)  GS=13 ;;
        16384) GS=14 ;;
        32768) GS=15 ;;
        65536) GS=16 ;;
    esac
    echo "Submitting BSQ gs${GS} (≈cb${CB}) (ImageNet)..."
    sbatch $SBATCH_COMMON \
        --job-name="bsq_full_${CB}" \
        --output="logs/slurm/bsq_full_cb${CB}_%j.out" \
        --error="logs/slurm/bsq_full_cb${CB}_%j.err" \
        --wrap="
$PREAMBLE
python3 scripts/train.py --model bsq $COMMON_ARGS \
    --bsq-beta 0.25 \
    --bsq-gamma0 1.0 \
    --bsq-gamma 1.0 \
    --bsq-zeta 0.1 \
    --bsq-group-size $GS \
    --bsq-l2-norm \
    --bsq-inv-temperature 1.0
"
    COUNT=$((COUNT + 1))

    echo "---"
done

echo ""
echo "Submitted $COUNT jobs (9 models x 5 codebook sizes) on full ImageNet with perceptual loss."
echo "Monitor with: squeue -u \$USER"

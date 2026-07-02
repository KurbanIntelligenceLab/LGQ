#!/bin/bash
# Train all non-LMB quantization models at 256-res with cb8192
# VQ, FSQ, LFQ, SimVQ
#
# Usage: bash run_train_256_all.sh

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
    --num-workers 0"

SBATCH_COMMON="--partition=gpu -A r00432 --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2-00:00:00"

PREAMBLE="module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH=\"\${PWD}:\${PWD}/external:\${PYTHONPATH:-}\"
export PYTHONFAULTHANDLER=1"

# VQ
echo "Submitting VQ 256-res cb8192..."
sbatch $SBATCH_COMMON \
    --job-name="vq_256_8k" \
    --output="logs/slurm/vq_256_8k_%j.out" \
    --error="logs/slurm/vq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model vq $COMMON_ARGS \
    --codebook-size 8192 \
    --commitment-weight 1.0 \
    --decay 0.8
"

# FSQ
echo "Submitting FSQ 256-res cb8192..."
sbatch $SBATCH_COMMON \
    --job-name="fsq_256_8k" \
    --output="logs/slurm/fsq_256_8k_%j.out" \
    --error="logs/slurm/fsq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model fsq $COMMON_ARGS \
    --levels 8 8 8 8 2
"

# LFQ
echo "Submitting LFQ 256-res cb8192..."
sbatch $SBATCH_COMMON \
    --job-name="lfq_256_8k" \
    --output="logs/slurm/lfq_256_8k_%j.out" \
    --error="logs/slurm/lfq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model lfq $COMMON_ARGS \
    --codebook-size 8192 \
    --entropy-loss-weight 0.1 \
    --diversity-gamma 1.0 \
    --spherical
"

# SimVQ
echo "Submitting SimVQ 256-res cb8192..."
sbatch $SBATCH_COMMON \
    --job-name="simvq_256_8k" \
    --output="logs/slurm/simvq_256_8k_%j.out" \
    --error="logs/slurm/simvq_256_8k_%j.err" \
    --wrap="
$PREAMBLE
python3 scripts/train.py --model sim_vq $COMMON_ARGS \
    --codebook-size 8192 \
    --use-mlp \
    --commitment-weight 10.0
"

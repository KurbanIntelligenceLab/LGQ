#!/bin/bash
# MaskGIT transformer training on top of frozen LMB tokenizer.
# Uses live latest_model.pt from canonical run (currently ep-23).
#
# Resumes via --resume latest if relaunched.
# Uses --num-workers 8 (matches SoftVQ/SimVQ launchers). The original
# segfault attributed to fork+JAX in 8910708 was actually the CUDA
# encoder probe in vq_adapter.py — fixed in d401071 (CPU probe).
#
#SBATCH --job-name=maskgit_lmb_cb16384
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
VQ_CKPT="results/lmb/lmb_abl1_ddp4_cb16384_s1234/checkpoints/latest_model.pt"
OUT_DIR="results/maskgit/lmb_cb16384_ep23_s1234_classcond"
RESUME="${RESUME:-}"

echo "=== MaskGIT on LMB cb16384 (live latest, ep-23) ==="
nvidia-smi -L || true

python -u MaskGIT-pytorch-main/train_maskgit_custom.py \
    --quantizer lmb \
    --vq-checkpoint "$VQ_CKPT" \
    --dataset-path "$DATA_ROOT/train" \
    --val-dataset-path "$DATA_ROOT/val" \
    --image-size 256 \
    --num-codebook-vectors 16384 \
    --num-classes 1000 \
    --batch-size 32 \
    --accum-grad 4 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --n-layers 12 \
    --dim 512 \
    --hidden-dim 2048 \
    --num-workers 8 \
    --sample-every 5 \
    --save-every 5 \
    --output-dir "$OUT_DIR" \
    ${RESUME:+--resume "$RESUME"}

echo "=== done ==="

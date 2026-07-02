#!/bin/bash
# rFID eval against the LATEST checkpoint of softvq_matched_cb${K}_256_ddp4_gb32_s1234.
# Pass codebook size via env: K=16384 sbatch ... eval_rfid_softvq_latest.sh
#
#SBATCH --job-name=eval_rfid_softvq
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --exclude=g36
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
: "${K:?Set K=<codebook_size>}"
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

export TORCH_CUDNN_V8_API_DISABLED=1

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
RUN_DIR="results/softvq/softvq_matched_cb${K}_256_ddp4_gb32_s1234"
CKPT="${RUN_DIR}/checkpoints/latest_model.pt"
OUT_CSV="${RUN_DIR}/rfid_eval.csv"

echo "=== rFID eval softvq K=${K}: $CKPT ==="
nvidia-smi -L || true

python scripts/evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --batch-size 32 \
    --num-workers 0 \
    --fid-samples 10000 \
    --write-eval-csv "$OUT_CSV"

echo "=== done softvq K=${K} ==="

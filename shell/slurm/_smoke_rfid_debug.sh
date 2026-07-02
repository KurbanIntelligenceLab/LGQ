#!/bin/bash
# Tiny rFID smoke job for debugging the segfault. faulthandler is enabled in
# evaluate.py so a segfault prints a Python stack to stderr.
#
#SBATCH --job-name=rfid_dbg
#SBATCH --partition=hopper
#SBATCH -A r00432
#SBATCH --qos=hopper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=00:15:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/%x_%j.err

set -e
module load python/gpu/3.11.5
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

DATA_ROOT="/N/project/de_briujn_graph/Projects/vector-quantize/Data/imagenet"
CKPT="results/lmb/_rfid_snapshots_2026-04-26/cb16384_latest.pt"
OUT_CSV="results/lmb/_rfid_snapshots_2026-04-26/cb16384_rfid_smoke.csv"

echo "=== smoke rFID DEBUG: $CKPT ==="
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0))"

python -u scripts/evaluation/evaluate.py \
    --checkpoint "$CKPT" \
    --data-root "$DATA_ROOT" \
    --split val \
    --batch-size 16 \
    --num-workers 0 \
    --num-samples 256 \
    --fid-samples 256 \
    --write-eval-csv "$OUT_CSV"

echo "=== smoke done ==="

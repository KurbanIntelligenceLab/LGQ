#!/bin/bash
# Start LMB experiment when a GPU becomes available

set -e

cd "$(dirname "$0")/.." || exit 1

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate main

export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

LMB_DIR="results/lmb/lmb_nb128_tau1.0-0.1_bs64_lr3e-4_dim128_20260116_013402_4a8d"
CONFIG_PATH="$LMB_DIR/config.json"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config not found at $CONFIG_PATH"
    exit 1
fi

# Read config
GPU=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['gpu'])")
RUN_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['run_name'])")
NUM_BINS=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['num_bins'])")
BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['batch_size'])")
LR=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['lr'])")
DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['dim'])")
EMBEDDING_DIM=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['embedding_dim'])")
EPOCHS=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['epochs'])")
IMAGE_SIZE=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['image_size'])")
NUM_WORKERS=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['num_workers'])")
SEED=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['seed'])")
DATA_ROOT=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['data_root'])")

echo "Starting LMB experiment..."
echo "GPU: $GPU"
echo "Run name: $RUN_NAME"
echo ""

# Check if GPU is available
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits --id=$GPU)
if [ "$GPU_UTIL" -gt 10 ]; then
    echo "Warning: GPU $GPU is currently $GPU_UTIL% utilized."
    echo "The training will start but may be slower due to GPU sharing."
    echo ""
fi

python3 scripts/train.py \
    --model lmb \
    --resume latest \
    --gpu "$GPU" \
    --run-name "$RUN_NAME" \
    --output-dir "$LMB_DIR" \
    --num-bins "$NUM_BINS" \
    --batch-size "$BATCH_SIZE" \
    --lr "$LR" \
    --dim "$DIM" \
    --embedding-dim "$EMBEDDING_DIM" \
    --epochs "$EPOCHS" \
    --image-size "$IMAGE_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --seed "$SEED" \
    --data-root "$DATA_ROOT" \
    > "$LMB_DIR/train.log" 2>&1 &

echo "LMB training started with PID: $!"
echo "Monitor with: tail -f $LMB_DIR/train.log"

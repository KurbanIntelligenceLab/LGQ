#!/bin/bash
# Resume all experiments from their latest checkpoints

set -e

cd "$(dirname "$0")/.." || exit 1

source /opt/miniforge3/etc/profile.d/conda.sh
conda activate main

export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

echo "Resuming all experiments..."
echo "=" | head -c 80 && echo ""

# FSQ (GPU 0)
echo "Resuming FSQ on GPU 0..."
nohup python3 scripts/train.py \
    --model fsq \
    --resume latest \
    --gpu 0 \
    --run-name fsq_lv8-5-5-5_bs32_lr3e-4_dim128_20260122_002041_5b2b \
    --output-dir results/fsq/fsq_lv8-5-5-5_bs32_lr3e-4_dim128_20260122_002041_5b2b \
    --levels 8 5 5 5 \
    --batch-size 32 \
    --lr 0.0003 \
    --dim 128 \
    --embedding-dim 128 \
    --epochs 100 \
    --image-size 128 \
    --num-workers 8 \
    --seed 1234 \
    > results/fsq/fsq_lv8-5-5-5_bs32_lr3e-4_dim128_20260122_002041_5b2b/train.log 2>&1 &
echo "  PID: $!"

# VQ (GPU 1)
echo "Resuming VQ on GPU 1..."
nohup python3 scripts/train.py \
    --model vq \
    --resume latest \
    --gpu 1 \
    --run-name vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0 \
    --output-dir results/vq/vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0 \
    --codebook-size 16384 \
    --batch-size 32 \
    --lr 0.0003 \
    --dim 128 \
    --embedding-dim 128 \
    --epochs 100 \
    --image-size 128 \
    --num-workers 8 \
    --seed 1234 \
    > results/vq/vq_cb16384_bs32_lr3e-4_dim128_20260122_002040_25e0/train.log 2>&1 &
echo "  PID: $!"

# SIM_VQ (GPU 2)
echo "Resuming SIM_VQ on GPU 2..."
nohup python3 scripts/train.py \
    --model sim_vq \
    --resume latest \
    --gpu 2 \
    --run-name sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32 \
    --output-dir results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32 \
    --codebook-size 16384 \
    --batch-size 32 \
    --lr 0.0003 \
    --dim 128 \
    --embedding-dim 128 \
    --epochs 100 \
    --image-size 128 \
    --num-workers 8 \
    --seed 1234 \
    > results/sim_vq/sim_vq_cb16384_bs32_lr3e-4_dim128_20260122_002042_1d32/train.log 2>&1 &
echo "  PID: $!"

# LMB-Fixed (GPU 3)
echo "Resuming LMB-Fixed on GPU 3..."
nohup python3 scripts/train.py \
    --model lmb \
    --resume latest \
    --gpu 3 \
    --run-name lmb_fixed_init \
    --output-dir results/lmb/lmb_fixed_init \
    --num-bins 16384 \
    --flatten-channels \
    --batch-size 32 \
    --lr 0.0003 \
    --dim 128 \
    --embedding-dim 128 \
    --epochs 100 \
    --image-size 128 \
    --num-workers 8 \
    --seed 1234 \
    > results/lmb/lmb_fixed_init/train.log 2>&1 &
echo "  PID: $!"

# LMB (GPU 2 - will wait if SIM_VQ is running, or use a different GPU)
# Note: LMB and SIM_VQ both want GPU 2. Check if GPU 2 is available first.
echo "Resuming LMB on GPU 2 (may conflict with SIM_VQ)..."
nohup python3 scripts/train.py \
    --model lmb \
    --resume latest \
    --gpu 2 \
    --run-name lmb_nb128_tau1.0-0.1_bs64_lr3e-4_dim128_20260116_013402_4a8d \
    --output-dir results/lmb/lmb_nb128_tau1.0-0.1_bs64_lr3e-4_dim128_20260116_013402_4a8d \
    --num-bins 128 \
    --batch-size 64 \
    --lr 0.0003 \
    --dim 128 \
    --embedding-dim 128 \
    --epochs 100 \
    --image-size 128 \
    --num-workers 8 \
    --seed 1234 \
    > results/lmb/lmb_nb128_tau1.0-0.1_bs64_lr3e-4_dim128_20260116_013402_4a8d/train.log 2>&1 &
echo "  PID: $!"

echo ""
echo "All experiments resumed!"
echo "Check logs in: results/*/train.log"
echo "Monitor with: tail -f results/*/train.log"

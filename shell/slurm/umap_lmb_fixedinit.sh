#!/bin/bash
#SBATCH --job-name=umap_lmb_fi
#SBATCH --partition=gpu
#SBATCH -A r00432
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-01:00:00
#SBATCH --output=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/umap_lmb_fixedinit_%j.out
#SBATCH --error=/N/project/de_briujn_graph/Projects/vector-quantize/logs/slurm/umap_lmb_fixedinit_%j.err

set -e
module load python/gpu/3.11.5
export PYTHONNOUSERSITE=1
source /N/project/de_briujn_graph/venvs/vqvae/bin/activate
cd /N/project/de_briujn_graph/Projects/vector-quantize
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"
export PYTHONFAULTHANDLER=1

echo "Node: $(hostname)  GPU: $CUDA_VISIBLE_DEVICES"
python3 --version

python3 scripts/plotting/umap_lmb_fixedinit.py \
    --checkpoint results/lmb/lmb_abl1_ddp4_cb16384_s1234_fixedinit/checkpoints/best_model.pt \
    --data-root Data/imagenet \
    --num-images 1000 \
    --subsample-latents 15000 \
    --output-dir plots/lmb_fixedinit

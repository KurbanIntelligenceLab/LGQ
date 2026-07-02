#!/bin/bash
# Run any command on GPU 1 only (only GPU 1 is visible as cuda:0).
# Usage: ./shell/run_on_gpu1.sh [command and args...]
# Example: ./shell/run_on_gpu1.sh ./shell/train_fsq.sh
# Example: ./shell/run_on_gpu1.sh python scripts/evaluate.py --checkpoint results/fsq/...
# To use GPU 1 in your current shell: export CUDA_VISIBLE_DEVICES=1

export CUDA_VISIBLE_DEVICES=1
if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 ./shell/train_fsq.sh"
    exit 1
fi
exec "$@"

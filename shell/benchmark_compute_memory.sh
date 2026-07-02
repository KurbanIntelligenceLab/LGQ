#!/bin/bash
# Run wall-clock and memory benchmark across all quantization models.
# Writes results/compute_memory_benchmark.json and results/compute_memory_benchmark.txt

set -e

cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="${PWD}:${PWD}/external:${PYTHONPATH:-}"

BATCH_SIZE="${BATCH_SIZE:-64}"
STEPS="${STEPS:-50}"
WARMUP="${WARMUP:-5}"
# Use GPU by default; set NO_CUDA=1 or --no-cuda for CPU-only
EXTRA_ARGS=("${@}")

python3 scripts/benchmark_compute_memory.py \
  --batch-size "$BATCH_SIZE" \
  --warmup "$WARMUP" \
  --steps "$STEPS" \
  "${EXTRA_ARGS[@]}"

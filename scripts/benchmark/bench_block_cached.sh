#!/bin/bash
# Block-cached MARS vs AR benchmark (Table 5)
# Tests block sizes 4, 8, 16, 32 at batch_size=16

BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

MARS_MODEL="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk4/checkpoint-final"
AR_MODEL="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/ar_sft/checkpoint-final"

python scripts/benchmark/bench_block_cached.py \
    --mars_model "${MARS_MODEL}" \
    --ar_model "${AR_MODEL}" \
    --batch_size 16 \
    --block_sizes "4,8,16,32" \
    --threshold 0.95 \
    --limit 200 \
    --output "results/block_cached_vs_ar.json"

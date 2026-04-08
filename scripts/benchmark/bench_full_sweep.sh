#!/bin/bash
# Full parameter sweep: block_size x batch_size (Figure 5)
# Tests block_sizes {1,2,4,8,16,32,64,128,256} x batch_sizes {1,2,4,8,16,32}

BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"

# 0.5B sweep
python scripts/benchmark/bench_full_sweep.py \
    --mars_model "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk4/checkpoint-final" \
    --ar_model "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/ar_sft/checkpoint-final" \
    --block_sizes "1,2,4,8,16,32,64,128,256" \
    --batch_sizes "1,2,4,8,16,32" \
    --threshold 0.95 \
    --limit 256 \
    --model_name "0.5B" \
    --output "results/sweep_0.5b.csv"

# 7B sweep (if models exist)
MARS_7B="${BASE_DIR}/models/Qwen2.5-7B-Instruct/mars_blk4/checkpoint-final"
AR_7B="${BASE_DIR}/models/Qwen2.5-7B-Instruct/ar_sft/checkpoint-final"
if [ -d "${MARS_7B}" ] && [ -d "${AR_7B}" ]; then
    python scripts/benchmark/bench_full_sweep.py \
        --mars_model "${MARS_7B}" \
        --ar_model "${AR_7B}" \
        --block_sizes "1,2,4,8,16,32,64,128,256" \
        --batch_sizes "1,2,4,8,16,32" \
        --threshold 0.95 \
        --limit 256 \
        --model_name "7B" \
        --output "results/sweep_7b.csv"
fi

#!/bin/bash
# Acceptance metric sensitivity sweep (Appendix Figure 7)
# Tests entropy and margin thresholds on GSM8K with MARS blk4
# Probability sweep is covered by eval_threshold_sweep.sh

BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

NUM_GPUS=8
RESULTS="${BASE_DIR}/results/acceptance_metric_sweep"

# MARS blk4 (with SFT loss)
MARS_BLK4="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk4/checkpoint-final"

# ==========================================
# Entropy sweep: H(p) <= threshold
# ==========================================
echo "=========================================="
echo ">>> Entropy threshold sweep"
echo "=========================================="

ENTROPY_THRESHOLDS="0.25 0.3 0.35 0.4 0.45"

for thresh in $ENTROPY_THRESHOLDS; do
    echo "=========================================="
    echo ">>> Entropy threshold = ${thresh}"
    echo "=========================================="
    accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
        --tasks gsm8k_cot --num_fewshot 0 \
        --model mars --apply_chat_template \
        --output_path "${RESULTS}/entropy_${thresh}" \
        --log_samples \
        --trust_remote_code \
        --confirm_run_unsafe_code \
        --model_args "pretrained=${MARS_BLK4},max_new_tokens=256,steps=256,block_size=4,cfg=0.0,right_shift_logits=True,confidence_threshold=${thresh},acceptance_metric=entropy"
done

# ==========================================
# Top-2 margin sweep: P(top1) - P(top2) >= threshold
# ==========================================
echo "=========================================="
echo ">>> Margin threshold sweep"
echo "=========================================="

MARGIN_THRESHOLDS="0.5 0.6 0.7 0.8 0.9 0.95"

for thresh in $MARGIN_THRESHOLDS; do
    echo "=========================================="
    echo ">>> Margin threshold = ${thresh}"
    echo "=========================================="
    accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
        --tasks gsm8k_cot --num_fewshot 0 \
        --model mars --apply_chat_template \
        --output_path "${RESULTS}/margin_${thresh}" \
        --log_samples \
        --trust_remote_code \
        --confirm_run_unsafe_code \
        --model_args "pretrained=${MARS_BLK4},max_new_tokens=256,steps=256,block_size=4,cfg=0.0,right_shift_logits=True,confidence_threshold=${thresh},acceptance_metric=margin"
done

echo "=========================================="
echo ">>> All done! Results in ${RESULTS}/"
echo "=========================================="

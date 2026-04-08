#!/bin/bash
# Full threshold sweep for Pareto analysis (Appendix Table 15)
# Sweeps tau in {0.95, 0.9, 0.8, 0.7, 0.6, 0.5} on GSM8K
# Part 1: MARS (with SFT loss) blk{4,8,16}
# Part 2: MARS (no SFT loss) blk{4,8,16}

BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

NUM_GPUS=8
RESULTS="${BASE_DIR}/results/threshold_sweep"

THRESHOLDS="0.95 0.9 0.8 0.7 0.6 0.5"

# ============================================
# Part 1: MARS (with SFT loss) threshold sweep
# ============================================
echo "============================================"
echo "=== Part 1: MARS (with SFT loss) sweep   ==="
echo "============================================"

mars_blk4="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk4/checkpoint-final"
mars_blk8="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk8/checkpoint-final"
mars_blk16="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk16/checkpoint-final"

for blk in 4 8 16; do
    eval model_path=\$mars_blk${blk}
    echo ">>> MARS (with SFT) block_size=${blk}"

    for thresh in $THRESHOLDS; do
        echo ">>> MARS blk=${blk}, threshold=${thresh}"
        accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
            --tasks gsm8k_cot --num_fewshot 0 \
            --model mars --apply_chat_template \
            --output_path "${RESULTS}/mars/blk${blk}/thresh_${thresh}" \
            --log_samples \
            --model_args "pretrained=${model_path},max_new_tokens=256,steps=256,block_size=${blk},cfg=0.0,right_shift_logits=True,confidence_threshold=${thresh}"
    done
done

# ============================================
# Part 2: MARS (no SFT loss) threshold sweep
# ============================================
echo "============================================"
echo "=== Part 2: MARS (no SFT loss) sweep     ==="
echo "============================================"

mars_no_sft_blk4="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk4/checkpoint-final"
mars_no_sft_blk8="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk8/checkpoint-final"
mars_no_sft_blk16="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk16/checkpoint-final"

for blk in 4 8 16; do
    eval model_path=\$mars_no_sft_blk${blk}
    echo ">>> MARS (no SFT) block_size=${blk}"

    for thresh in $THRESHOLDS; do
        echo ">>> MARS (no SFT) blk=${blk}, threshold=${thresh}"
        accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
            --tasks gsm8k_cot --num_fewshot 0 \
            --model mars --apply_chat_template \
            --output_path "${RESULTS}/mars_no_sft/blk${blk}/thresh_${thresh}" \
            --log_samples \
            --model_args "pretrained=${model_path},max_new_tokens=256,steps=256,block_size=${blk},cfg=0.0,right_shift_logits=True,confidence_threshold=${thresh}"
    done
done

echo "=========================================="
echo ">>> All done! Results in ${RESULTS}/"
echo "=========================================="

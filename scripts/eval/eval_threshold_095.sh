#!/bin/bash
# Multi-token evaluation at threshold=0.95 (Table 4)
# 0.5B: 6 MARS models (3 block sizes x with/without SFT loss)
# 7B: MARS blk4
# Tasks: IFEval, BBH, MMLU-Pro, GPQA, GSM8K(0-shot), HumanEval
# Records avg_tokens_per_forward via sampler stats

BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

NUM_GPUS=8
RESULTS="${BASE_DIR}/results/threshold_095"

TASKS=(
    "ifeval 0"
    "bbh 3"
    "mmlu_pro 0"
    "gpqa_main_cot_zeroshot 0"
    "gsm8k_cot 0"
    "humaneval_instruct 0"
)

eval_mars() {
    local model_path=$1 model_name=$2 blk=$3 results_dir=$4
    for task_spec in "${TASKS[@]}"; do
        read -r task fewshot <<< "$task_spec"
        extra_args=""
        if [ "$task" = "humaneval_instruct" ]; then
            extra_args="--confirm_run_unsafe_code"
        fi
        echo ">>> ${model_name}: ${task} (${fewshot}-shot) threshold=0.95"
        accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
            --tasks "${task}" --num_fewshot "${fewshot}" \
            --model mars --apply_chat_template \
            --output_path "${results_dir}/${model_name}/${task}_${fewshot}shot" \
            --log_samples \
            --batch_size 1 \
            --model_args "pretrained=${model_path},max_new_tokens=256,steps=256,block_size=${blk},cfg=0.0,right_shift_logits=True,confidence_threshold=0.95" \
            ${extra_args}
    done
}

# ============================================
# 0.5B models at threshold=0.95
# ============================================
echo "=========================================="
echo ">>> 0.5B models at threshold=0.95"
echo "=========================================="

# MARS (with SFT loss) blk4
eval_mars \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk4/checkpoint-final" \
    "0.5b_mars_blk4" 4 "${RESULTS}"

# MARS (with SFT loss) blk8
eval_mars \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk8/checkpoint-final" \
    "0.5b_mars_blk8" 8 "${RESULTS}"

# MARS (with SFT loss) blk16
eval_mars \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk16/checkpoint-final" \
    "0.5b_mars_blk16" 16 "${RESULTS}"

# MARS (no SFT loss) blk4
eval_mars \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk4/checkpoint-final" \
    "0.5b_mars_no_sft_blk4" 4 "${RESULTS}"

# MARS (no SFT loss) blk8
eval_mars \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk8/checkpoint-final" \
    "0.5b_mars_no_sft_blk8" 8 "${RESULTS}"

# MARS (no SFT loss) blk16
eval_mars \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk16/checkpoint-final" \
    "0.5b_mars_no_sft_blk16" 16 "${RESULTS}"

# ============================================
# 7B models at threshold=0.95
# ============================================
echo "=========================================="
echo ">>> 7B models at threshold=0.95"
echo "=========================================="

MARS_7B="${BASE_DIR}/models/Qwen2.5-7B-Instruct/mars_blk4/checkpoint-final"

if [ -d "${MARS_7B}" ]; then
    eval_mars "${MARS_7B}" "7b_mars_blk4" 4 "${RESULTS}"
else
    echo ">>> 7B MARS model not found at ${MARS_7B}, skipping."
fi

echo ">>> Done: ${RESULTS}/"

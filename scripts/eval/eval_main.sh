#!/bin/bash
# Main evaluation: 6 MARS models x 7 tasks (Table 2)
# All at tau=1.0 (one token per forward pass)
# Reproduces: Table 2 main results

BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

NUM_GPUS=8
RESULTS="${BASE_DIR}/results/main"

# Model paths (adjust to your checkpoint locations)
declare -A MODELS
MODELS[mars_no_sft_blk4]="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk4/checkpoint-final"
MODELS[mars_no_sft_blk8]="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk8/checkpoint-final"
MODELS[mars_no_sft_blk16]="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk16/checkpoint-final"
MODELS[mars_blk4]="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk4/checkpoint-final"
MODELS[mars_blk8]="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk8/checkpoint-final"
MODELS[mars_blk16]="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk16/checkpoint-final"

declare -A BLOCK_SIZES
BLOCK_SIZES[mars_no_sft_blk4]=4
BLOCK_SIZES[mars_no_sft_blk8]=8
BLOCK_SIZES[mars_no_sft_blk16]=16
BLOCK_SIZES[mars_blk4]=4
BLOCK_SIZES[mars_blk8]=8
BLOCK_SIZES[mars_blk16]=16

TASKS=(
    "ifeval 0"
    "bbh 3"
    "mmlu_pro 0"
    "gpqa_main_cot_zeroshot 0"
    "gsm8k_cot 0"
    "gsm8k_cot 5"
    "humaneval_instruct 0"
)

for model_name in "mars_no_sft_blk4" "mars_no_sft_blk8" "mars_no_sft_blk16" "mars_blk4" "mars_blk8" "mars_blk16"; do
    model_path="${MODELS[$model_name]}"
    blk="${BLOCK_SIZES[$model_name]}"

    echo ">>> Model: ${model_name} (block_size=${blk})"

    for task_spec in "${TASKS[@]}"; do
        read -r task fewshot <<< "$task_spec"

        extra_args=""
        [ "$task" = "humaneval_instruct" ] && extra_args="--confirm_run_unsafe_code"

        echo ">>> ${model_name}: ${task} (${fewshot}-shot)"
        accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
            --tasks "${task}" --num_fewshot "${fewshot}" \
            --model mars --apply_chat_template \
            --output_path "${RESULTS}/${model_name}/${task}_${fewshot}shot" \
            --log_samples \
            --batch_size 1 \
            --model_args "pretrained=${model_path},max_new_tokens=256,steps=256,block_size=${blk},cfg=0.0,right_shift_logits=True" \
            ${extra_args}
    done
done

echo ">>> Done: ${RESULTS}/"

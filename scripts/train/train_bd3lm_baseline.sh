#!/bin/bash
# BD3LM Baseline Training + Evaluation (0.5B)
# Trains BD3LM from the AR SFT checkpoint, then evaluates on 6 benchmarks.
# Uses the dllm BD3LM trainer directly for comparison against MARS.
#
# Reproduces: Table 2 (BD3LM baseline row)
# Hardware: 8x H200 GPUs
# Total batch size: 48 * 1 * 8 = 384

# === Configuration ===
BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"  # Set this!
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

NUM_GPUS=8
BLOCK_SIZE=4
AR_CHECKPOINT="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/ar_sft/checkpoint-final"
BD3LM_OUTPUT="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/bd3lm_blk${BLOCK_SIZE}"
RESULTS_DIR="${BASE_DIR}/results/qwen2.5_0.5b"

TASKS=(
    "ifeval 0"
    "bbh 3"
    "mmlu_pro 0"
    "gpqa_main_cot_zeroshot 0"
    "gsm8k_cot 0"
    "humaneval_instruct 0"
)

# Helper: eval BD3LM model
eval_bd3lm() {
    local task=$1 fewshot=$2
    extra_args=""
    [ "$task" = "humaneval_instruct" ] && extra_args="--confirm_run_unsafe_code"
    echo ">>> BD3LM blk=${BLOCK_SIZE}: ${task} (${fewshot}-shot)"
    accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
        --tasks "${task}" --num_fewshot "${fewshot}" \
        --model bd3lm --apply_chat_template \
        --output_path "${RESULTS_DIR}/bd3lm_blk${BLOCK_SIZE}" \
        --log_samples \
        --batch_size 1 \
        --model_args "pretrained=${BD3LM_OUTPUT}/checkpoint-final,max_new_tokens=256,steps=256,block_size=${BLOCK_SIZE},cfg=0.0,right_shift_logits=True" \
        ${extra_args}
}

# [1/2] Train BD3LM
echo "=========================================="
echo ">>> [1/2] Train BD3LM (block_size=${BLOCK_SIZE})"
echo "=========================================="
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes ${NUM_GPUS} \
    examples/a2d/bd3lm/sft.py \
    --model_name_or_path "${AR_CHECKPOINT}" \
    --output_dir "${BD3LM_OUTPUT}" \
    --dataset_args "allenai/Dolci-Instruct-SFT" \
    --num_train_epochs 5 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --max_length 512 \
    --save_total_limit 3 \
    --block_size ${BLOCK_SIZE} \
    --right_shift_logits \
    --save_steps 1000 \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --group_by_length \
    --bf16 \
    --num_proc 64

# [2/2] Eval BD3LM
echo "=========================================="
echo ">>> [2/2] Eval BD3LM (block_size=${BLOCK_SIZE})"
echo "=========================================="
for task_spec in "${TASKS[@]}"; do
    read -r task fewshot <<< "$task_spec"
    eval_bd3lm "${task}" "${fewshot}"
done

echo ">>> All done! Results in ${RESULTS_DIR}/"

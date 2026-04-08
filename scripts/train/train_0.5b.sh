#!/bin/bash
# MARS 0.5B Training Pipeline
# Stage 1: AR SFT (5 epochs)
# Stage 2: MARS with SFT loss (block_size=4, 5 epochs)
# Stage 3: Evaluate both on 6 benchmarks
#
# Reproduces: Table 2 (AR SFT row + MARS blk4 row)
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
BASE_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
AR_OUTPUT="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/ar_sft"
MARS_OUTPUT="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk${BLOCK_SIZE}"
RESULTS_DIR="${BASE_DIR}/results/qwen2.5_0.5b"

TASKS=(
    "ifeval 0"
    "bbh 3"
    "mmlu_pro 0"
    "gpqa_main_cot_zeroshot 0"
    "gsm8k_cot 0"
    "humaneval_instruct 0"
)

# Helper: eval AR model
eval_ar() {
    local task=$1 fewshot=$2
    extra_args=""
    [ "$task" = "humaneval_instruct" ] && extra_args="--confirm_run_unsafe_code"
    echo ">>> AR SFT: ${task} (${fewshot}-shot)"
    accelerate launch --num_processes "${NUM_GPUS}" -m lm_eval --model hf \
        --tasks "${task}" --num_fewshot "${fewshot}" \
        --apply_chat_template \
        --output_path "${RESULTS_DIR}/ar_sft" \
        --log_samples \
        --model_args "pretrained=${AR_OUTPUT}/checkpoint-final,dtype=bfloat16" \
        --gen_kwargs "max_gen_toks=256" \
        --batch_size 4 \
        ${extra_args}
}

# Helper: eval MARS model
eval_mars() {
    local task=$1 fewshot=$2
    extra_args=""
    [ "$task" = "humaneval_instruct" ] && extra_args="--confirm_run_unsafe_code"
    echo ">>> MARS blk=${BLOCK_SIZE}: ${task} (${fewshot}-shot)"
    accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
        --tasks "${task}" --num_fewshot "${fewshot}" \
        --model full_mask --apply_chat_template \
        --output_path "${RESULTS_DIR}/mars_blk${BLOCK_SIZE}" \
        --log_samples \
        --batch_size 1 \
        --model_args "pretrained=${MARS_OUTPUT}/checkpoint-final,max_new_tokens=256,steps=256,block_size=${BLOCK_SIZE},cfg=0.0,right_shift_logits=True" \
        ${extra_args}
}

# [1/4] Train AR SFT
echo "=========================================="
echo ">>> [1/4] Train AR SFT"
echo "=========================================="
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes ${NUM_GPUS} \
    train/train_ar_sft.py \
    --model_name_or_path "${BASE_MODEL}" \
    --output_dir "${AR_OUTPUT}" \
    --dataset_args "allenai/Dolci-Instruct-SFT" \
    --num_train_epochs 5 \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 48 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --max_length 512 \
    --save_total_limit 3 \
    --save_steps 1000 \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --group_by_length \
    --bf16 \
    --num_proc 64

# [2/4] Eval AR SFT
echo "=========================================="
echo ">>> [2/4] Eval AR SFT"
echo "=========================================="
for task_spec in "${TASKS[@]}"; do
    read -r task fewshot <<< "$task_spec"
    eval_ar "${task}" "${fewshot}"
done

# [3/4] Train MARS (blk4)
echo "=========================================="
echo ">>> [3/4] Train MARS (block_size=${BLOCK_SIZE})"
echo "=========================================="
accelerate launch \
    --config_file scripts/accelerate_configs/zero2.yaml \
    --num_processes ${NUM_GPUS} \
    train/train_mars.py \
    --model_name_or_path "${AR_OUTPUT}/checkpoint-final" \
    --output_dir "${MARS_OUTPUT}" \
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
    --ar_weight 1.0 \
    --save_steps 1000 \
    --logging_steps 10 \
    --warmup_ratio 0.03 \
    --group_by_length \
    --bf16 \
    --num_proc 64

# [4/4] Eval MARS
echo "=========================================="
echo ">>> [4/4] Eval MARS (block_size=${BLOCK_SIZE})"
echo "=========================================="
for task_spec in "${TASKS[@]}"; do
    read -r task fewshot <<< "$task_spec"
    eval_mars "${task}" "${fewshot}"
done

echo ">>> All done! Results in ${RESULTS_DIR}/"

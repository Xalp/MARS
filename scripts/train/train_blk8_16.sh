#!/bin/bash
# MARS 0.5B Block Size Ablation (blk8, blk16)
# Trains 4 models from the AR SFT checkpoint:
#   1. MARS (no SFT loss) blk8
#   2. MARS (no SFT loss) blk16
#   3. MARS (with SFT loss) blk8
#   4. MARS (with SFT loss) blk16
# Evaluates each on 6 benchmarks.
#
# Reproduces: Table 3 (block size ablation rows)
# Hardware: 8x H200 GPUs
# Total batch size: 48 * 1 * 8 = 384

# === Configuration ===
BASE_DIR="${BASE_DIR:-/path/to/experiment/root}"  # Set this!
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

export PYTHONPATH="${REPO_DIR}:${PYTHONPATH}"
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=True

NUM_GPUS=8
AR_CHECKPOINT="${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/ar_sft/checkpoint-final"
DATASET="allenai/Dolci-Instruct-SFT"
RESULTS_BASE="${BASE_DIR}/results/qwen2.5_0.5b"

TASKS=(
    "ifeval 0"
    "bbh 3"
    "mmlu_pro 0"
    "gpqa_main_cot_zeroshot 0"
    "gsm8k_cot 0"
    "humaneval_instruct 0"
)

# Helper: eval MARS model
eval_mars() {
    local model_path=$1 blk=$2 task=$3 fewshot=$4 results=$5
    extra_args=""
    [ "$task" = "humaneval_instruct" ] && extra_args="--confirm_run_unsafe_code"
    echo ">>> MARS blk=${blk}: ${task} (${fewshot}-shot)"
    accelerate launch --num_processes "${NUM_GPUS}" mars/eval_harness.py \
        --tasks "${task}" --num_fewshot "${fewshot}" \
        --model full_mask --apply_chat_template \
        --output_path "${results}" \
        --log_samples \
        --batch_size 1 \
        --model_args "pretrained=${model_path},max_new_tokens=256,steps=256,block_size=${blk},cfg=0.0,right_shift_logits=True" \
        ${extra_args}
}

# Helper: train + eval one configuration
train_and_eval() {
    local train_script=$1 output_dir=$2 blk=$3 label=$4 results_dir=$5
    shift 5
    local extra_train_args=("$@")

    echo "=========================================="
    echo ">>> Training: ${label} (block_size=${blk})"
    echo "=========================================="
    accelerate launch \
        --config_file scripts/accelerate_configs/zero2.yaml \
        --num_processes ${NUM_GPUS} \
        "${train_script}" \
        --model_name_or_path "${AR_CHECKPOINT}" \
        --output_dir "${output_dir}" \
        --dataset_args "${DATASET}" \
        --num_train_epochs 5 \
        --learning_rate 5e-6 \
        --per_device_train_batch_size 48 \
        --gradient_accumulation_steps 1 \
        --eval_strategy "no" \
        --max_length 512 \
        --save_total_limit 3 \
        --block_size ${blk} \
        --right_shift_logits \
        --save_steps 1000 \
        --logging_steps 10 \
        --warmup_ratio 0.03 \
        --group_by_length \
        --bf16 \
        --num_proc 64 \
        "${extra_train_args[@]}"

    echo "=========================================="
    echo ">>> Eval: ${label} (block_size=${blk})"
    echo "=========================================="
    local model_path="${output_dir}/checkpoint-final"
    for task_spec in "${TASKS[@]}"; do
        read -r task fewshot <<< "$task_spec"
        eval_mars "${model_path}" "${blk}" "${task}" "${fewshot}" "${results_dir}"
    done
}

# [1/4] MARS (no SFT loss) blk8
train_and_eval \
    train/train_mars_no_sft.py \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk8" \
    8 "MARS (no SFT)" \
    "${RESULTS_BASE}/mars_no_sft_blk8"

# [2/4] MARS (no SFT loss) blk16
train_and_eval \
    train/train_mars_no_sft.py \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_no_sft_blk16" \
    16 "MARS (no SFT)" \
    "${RESULTS_BASE}/mars_no_sft_blk16"

# [3/4] MARS (with SFT loss) blk8
train_and_eval \
    train/train_mars.py \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk8" \
    8 "MARS" \
    "${RESULTS_BASE}/mars_blk8" \
    --ar_weight 1.0

# [4/4] MARS (with SFT loss) blk16
train_and_eval \
    train/train_mars.py \
    "${BASE_DIR}/models/Qwen2.5-0.5B-Instruct/mars_blk16" \
    16 "MARS" \
    "${RESULTS_BASE}/mars_blk16" \
    --ar_weight 1.0

echo ">>> All done! Results in ${RESULTS_BASE}/"

#!/bin/bash
# Upload all trained MARS models to HuggingFace Hub.
# Usage: bash scripts/upload_models.sh
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login

set -e

BASE_DIR="${BASE_DIR:?Set BASE_DIR to your experiment root}"
HF_ORG="${HF_ORG:-Xalp}"  # HuggingFace org or username

# Model path -> HF repo name
declare -A MODELS

# 0.5B
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar/lr-5e-6-ep-5-seq512/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-AR-SFT"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-blk4"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk8/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-blk8"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk16/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-blk16"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-blk4-no-sft"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm/lr-5e-6-ep-5-seq512-blk8/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-blk8-no-sft"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm/lr-5e-6-ep-5-seq512-blk16/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-blk16-no-sft"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_bd3lm/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-0.5B-BD3LM-blk4"

# 7B
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-7B-Instruct/general_sft_ar/lr-5e-6-ep-5-seq512/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-7B-AR-SFT"
MODELS["${BASE_DIR}/models/a2d/Qwen2.5-7B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"]="${HF_ORG}/MARS-Qwen2.5-7B-blk4"

upload() {
    local path=$1 repo=$2
    if [ ! -d "$path" ]; then
        echo "[SKIP] $repo -- $path not found"
        return
    fi
    echo "[UPLOAD] $path -> $repo"
    huggingface-cli upload "$repo" "$path" . --repo-type model
}

echo "Uploading ${#MODELS[@]} models to HuggingFace..."
echo ""

for path in "${!MODELS[@]}"; do
    upload "$path" "${MODELS[$path]}"
    echo ""
done

echo "Done."

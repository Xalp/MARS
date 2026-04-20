#!/bin/bash
# Upload all trained MARS models to HuggingFace Hub.
# Usage: bash scripts/upload_models.sh
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login
#
# Uploads in the order declared below. 7B blk8/16 first (newest, highest
# priority), then 7B blk4 and 7B AR SFT, then all 0.5B models.

set -e

BASE_DIR="${BASE_DIR:?Set BASE_DIR to your experiment root}"
HF_ORG="${HF_ORG:-Xalp}"  # HuggingFace org or username

# Parallel ordered arrays so bash iterates in the order declared.
PATHS=()
REPOS=()
add() {
    PATHS+=("$1")
    REPOS+=("$2")
}

# ---- 7B (upload first) ----
add "${BASE_DIR}/models/a2d/Qwen2.5-7B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk8/checkpoint-final"  "${HF_ORG}/MARS-Qwen2.5-7B-blk8"
add "${BASE_DIR}/models/a2d/Qwen2.5-7B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk16/checkpoint-final" "${HF_ORG}/MARS-Qwen2.5-7B-blk16"
# add "${BASE_DIR}/models/a2d/Qwen2.5-7B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"  "${HF_ORG}/MARS-Qwen2.5-7B-blk4"
# add "${BASE_DIR}/models/a2d/Qwen2.5-7B-Instruct/general_sft_ar/lr-5e-6-ep-5-seq512/checkpoint-final"                   "${HF_ORG}/MARS-Qwen2.5-7B-AR-SFT"

# # ---- 0.5B ----
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar/lr-5e-6-ep-5-seq512/checkpoint-final"                    "${HF_ORG}/MARS-Qwen2.5-0.5B-AR-SFT"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"    "${HF_ORG}/MARS-Qwen2.5-0.5B-blk4"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk8/checkpoint-final"    "${HF_ORG}/MARS-Qwen2.5-0.5B-blk8"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm_ar/lr-5e-6-ep-5-seq512-blk16/checkpoint-final"   "${HF_ORG}/MARS-Qwen2.5-0.5B-blk16"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"       "${HF_ORG}/MARS-Qwen2.5-0.5B-blk4-no-sft"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm/lr-5e-6-ep-5-seq512-blk8/checkpoint-final"       "${HF_ORG}/MARS-Qwen2.5-0.5B-blk8-no-sft"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_fm/lr-5e-6-ep-5-seq512-blk16/checkpoint-final"      "${HF_ORG}/MARS-Qwen2.5-0.5B-blk16-no-sft"
# add "${BASE_DIR}/models/a2d/Qwen2.5-0.5B-Instruct/general_sft_ar_then_bd3lm/lr-5e-6-ep-5-seq512-blk4/checkpoint-final"    "${HF_ORG}/MARS-Qwen2.5-0.5B-BD3LM-blk4"

upload() {
    local path=$1 repo=$2
    if [ ! -d "$path" ]; then
        echo "[SKIP] $repo -- $path not found"
        return
    fi
    echo "[UPLOAD] $path -> $repo"
    huggingface-cli upload "$repo" "$path" . --repo-type model
}

echo "Uploading ${#PATHS[@]} models to HuggingFace (7B first)..."
echo ""

for i in "${!PATHS[@]}"; do
    upload "${PATHS[$i]}" "${REPOS[$i]}"
    echo ""
done

echo "Done."

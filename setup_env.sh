#!/bin/bash
# Create and configure the mars conda environment.

set -e

ENV_NAME=mars
PYTHON_VERSION=3.11

conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0 accelerate>=0.30.0 deepspeed>=0.14.0
pip install lm-eval>=0.4.0 datasets safetensors peft

echo "Done. Run 'conda activate ${ENV_NAME}' to use."

#!/bin/bash
# Create and configure the mars conda environment.

set -e

ENV_NAME=mars
PYTHON_VERSION=3.11

conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

pip install -r requirements.txt

echo "Done. Run 'conda activate ${ENV_NAME}' to use."

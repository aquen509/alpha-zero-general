#!/bin/bash
set -euo pipefail

PYTHON_VERSION=${1:-"3.14"}
VENV_PATH=${2:-".venv"}

if ! command -v uv >/dev/null 2>&1; then
    echo "The uv package manager is required. Please install it from https://github.com/astral-sh/uv."
    exit 1
fi

echo "Creating virtual environment with Python ${PYTHON_VERSION} at ${VENV_PATH}" 
uv venv --python "${PYTHON_VERSION}" "${VENV_PATH}"

source "${VENV_PATH}/bin/activate"
uv pip install -r requirements.txt

echo "Environment ready. Activate it with: source ${VENV_PATH}/bin/activate"

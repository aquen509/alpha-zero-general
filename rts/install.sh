#!/bin/bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "The uv package manager is required. Install it from https://github.com/astral-sh/uv"
    exit 1
fi

uv pip install numpy pygame torch==2.9.0

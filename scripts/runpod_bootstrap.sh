#!/usr/bin/env bash
# Run after extracting the prepared bundle under /workspace. No cloud resources created.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
workspace="${RUNPOD_WORKSPACE:-/workspace}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$workspace/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$workspace/.python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export HF_HOME="${HF_HOME:-$workspace/hf-cache}"
uv_bin="${UV_BIN:-$workspace/tools/uv}"
test -x "$uv_bin"
"$uv_bin" python install --no-bin 3.11.15
"$uv_bin" sync --locked --extra train --python 3.11.15
"$uv_bin" run --locked --extra train sim-student doctor
"$uv_bin" run --locked --extra train python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU visible:", torch.cuda.is_available())'

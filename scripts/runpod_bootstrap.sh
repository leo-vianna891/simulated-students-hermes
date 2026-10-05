#!/usr/bin/env bash
# Prepare this checkout under /workspace; creates no cloud resources.
set -euo pipefail
umask 077
cd "$(dirname "${BASH_SOURCE[0]}")/.."
workspace="${RUNPOD_WORKSPACE:-/workspace}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$workspace/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$workspace/.python}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export HF_HOME="${HF_HOME:-$workspace/hf-cache}"
uv_bin="${UV_BIN:-$workspace/tools/uv}"
if [[ ! -x "$uv_bin" ]]; then
  mkdir -p "$workspace" "$(dirname "$uv_bin")"
  temporary="$(mktemp -d "$workspace/.uv-install.XXXXXX")"
  trap 'rm -rf -- "$temporary"' EXIT
  curl --fail --location --silent --show-error --retry 3 \
    https://github.com/astral-sh/uv/releases/download/0.11.32/uv-x86_64-unknown-linux-gnu.tar.gz \
    --output "$temporary/uv.tar.gz"
  tar --no-same-owner --no-same-permissions -xzf "$temporary/uv.tar.gz" -C "$temporary"
  mv "$temporary/uv-x86_64-unknown-linux-gnu/uv" "$uv_bin"
  rm -rf -- "$temporary"
  trap - EXIT
fi
case "$("$uv_bin" --version)" in
  "uv 0.11.32 "*) ;;
  *) printf '%s\n' 'Provide uv 0.11.32 via UV_BIN.' >&2; exit 1 ;;
esac
"$uv_bin" python install --no-bin 3.11.15
"$uv_bin" sync --locked --extra train --python 3.11.15
"$uv_bin" run --locked --extra train sim-student doctor
"$uv_bin" run --locked --extra train python -c 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda, "GPU visible:", torch.cuda.is_available())'

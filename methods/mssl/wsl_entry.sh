#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$1"
shift
cd "$TASK_ROOT"
TASK_ENV="${POLYPBENCH_ENV:-$HOME/.venvs/polypbench-maskdino}"
source "$TASK_ENV/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$TASK_ROOT:$TASK_ROOT/third_party/MaskDINO:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
exec python "$@"

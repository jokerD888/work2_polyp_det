#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$1"
shift
cd "$TASK_ROOT"
TASK_ENV="${POLYPBENCH_ENV:-$HOME/.venvs/polypbench-maskdino}"
source "$TASK_ENV/bin/activate"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
exec python -m polypbench "$@"

#!/usr/bin/env bash
# Linux or WSL2. Does not install/modify the host NVIDIA driver.
set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_ENV="${POLYPBENCH_ENV:-$HOME/.venvs/polypbench-maskdino}"
TASK_CACHE="${POLYPBENCH_CACHE:-$HOME/.cache/polypbench}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export PATH="$CUDA_HOME/bin:$PATH"
export MAX_JOBS="${MAX_JOBS:-2}"
test -x "$CUDA_HOME/bin/nvcc" || { echo "CUDA 12.8 toolkit required: set CUDA_HOME" >&2; exit 1; }
command -v g++ >/dev/null
python3 -m venv "$TASK_ENV"
source "$TASK_ENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
# Minimal CUDA toolkit installations may omit cuSPARSE/cuBLAS headers. The pinned
# PyTorch wheels provide them; make those include directories visible to nvcc.
TASK_NVIDIA_INCLUDES="$(python -c 'import glob,site; print(":".join(p for base in site.getsitepackages() for p in glob.glob(base+"/nvidia/*/include")))')"
export CPATH="$TASK_NVIDIA_INCLUDES${CPATH:+:$CPATH}"
python -m pip install 'numpy<2' 'timm==0.6.13' 'opencv-python<4.12' scipy shapely h5py submitit scikit-image cython
python -m pip install -e "$TASK_ROOT"
cd "$TASK_ROOT"
python -m polypbench prepare-repo
mkdir -p "$TASK_CACHE"
TASK_D2="$TASK_CACHE/detectron2"
if [ ! -d "$TASK_D2" ]; then
  git clone https://github.com/facebookresearch/detectron2.git "$TASK_D2"
  git -C "$TASK_D2" checkout --detach 1e3e13bbf607b54f62205c4c33922521822fb298
fi
test "$(git -C "$TASK_D2" rev-parse HEAD)" = 1e3e13bbf607b54f62205c4c33922521822fb298 || { echo 'Detectron2 commit mismatch; existing checkout untouched' >&2; exit 1; }
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c 'import torch; major,minor=torch.cuda.get_device_capability(); print(f"{major}.{minor}")')}"
python -m pip install --no-build-isolation -e "$TASK_D2"
python -m pip install 'git+https://github.com/cocodataset/panopticapi.git@7bb4655548f98f3fedc07bf37e9040a992b054b0'
python -m pip install cityscapesScripts
python scripts/patch_maskdino_ops.py
cd third_party/MaskDINO/maskdino/modeling/pixel_decoder/ops
python -m pip install --no-build-isolation .
cd "$TASK_ROOT"
python -m polypbench doctor --config configs/maskdino_kvasir_official.yaml
python -m pip freeze > "$TASK_ENV/requirements.lock.txt"
echo "Environment ready: $TASK_ENV"

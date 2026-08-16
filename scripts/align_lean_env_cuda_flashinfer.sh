#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/lean/miniforge3}"
ENV_NAME="${ENV_NAME:-lean_env}"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"

if [ ! -x "$PY" ]; then
  echo "missing python: $PY" >&2
  exit 1
fi

echo "Using $PY"
"$PY" --version

echo
echo "==== current CUDA/FlashInfer package state ===="
"$PY" /mnt/e/python_project/scripts/report_cuda_package_versions.py || true

echo
echo "==== align CUDA packages ===="
# Rationale:
# - vLLM 0.24.0 pins flashinfer-python 0.6.12, which pulls cuda-tile[tileiras]
#   and naturally conflicts with torch 2.11.0's CUDA 13.0 toolkit family.
# - vLLM 0.22.1 still pins torch 2.11.0 but uses flashinfer-python 0.6.11.post2.
# - flashinfer-python 0.6.11.post2 depends on plain cuda-tile, so cuda-tile 1.1.0
#   can run with the CUDA 13.0 toolkit family without bypassing CCCL checks.
"$PY" -m pip install --upgrade \
  "torch==2.11.0" \
  "cuda-toolkit[all]==13.0.2" \
  "vllm==0.22.1" \
  "flashinfer-python==0.6.11.post2" \
  "flashinfer-cubin==0.6.11.post2" \
  "cuda-tile==1.1.0" \
  "setuptools==80.10.2" \
  "fsspec[http]==2026.4.0"

"$PY" -m pip uninstall -y \
  nvidia-cuda-tileiras \
  nvidia-cublas-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cudnn-cu12 \
  nvidia-cufft-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-cufile-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 || true

echo
echo "==== verify package consistency ===="
"$PY" -m pip check
"$PY" /mnt/e/python_project/scripts/report_cuda_package_versions.py

echo
echo "==== verify FlashInfer sampler import path ===="
ENV_ROOT="$(dirname "$(dirname "$PY")")"
export CUDA_HOME="${CUDA_HOME:-$ENV_ROOT/lib/python3.12/site-packages/nvidia/cu13}"
ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
ln -sfn "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
mkdir -p "$CUDA_HOME/lib/stubs"
if [ -f /usr/lib/wsl/lib/libcuda.so ]; then
  ln -sfn /usr/lib/wsl/lib/libcuda.so "$CUDA_HOME/lib/stubs/libcuda.so"
fi
SITE_PACKAGES="$ENV_ROOT/lib/python3.12/site-packages"
NVIDIA_LIB_DIRS="$(find "$SITE_PACKAGES/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib 2>/dev/null | paste -sd: -)"
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export LIBRARY_PATH="$CUDA_HOME/lib:$NVIDIA_LIB_DIRS:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$NVIDIA_LIB_DIRS:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
unset CCCL_DISABLE_CTK_COMPATIBILITY_CHECK
unset NVCC_PREPEND_FLAGS
unset VLLM_USE_FLASHINFER_SAMPLER
"$PY" - <<'PY'
import os
import torch

print(f"torch={torch.__version__}")
print(f"torch_cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"CUDA_HOME={os.environ.get('CUDA_HOME')}")
print("FlashInfer sampler will be enabled by vLLM default environment.")
PY

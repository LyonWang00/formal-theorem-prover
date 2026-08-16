#!/usr/bin/env bash

# Shared runtime environment for Lean SFT/vLLM workflows.
#
# Source this file after PY has been set. It makes the CUDA compiler installed
# by the Python CUDA packages visible to FlashInfer without bypassing version
# checks, and avoids inheriting Windows PATH entries inside WSL.

if [ -z "${PY:-}" ]; then
  echo "lean_env_runtime.sh requires PY to point to the Python executable" >&2
  return 2 2>/dev/null || exit 2
fi

PY_BIN="$(dirname "$PY")"
ENV_ROOT="$(dirname "$(dirname "$PY")")"

PY_VERSION="$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
SITE_PACKAGES="$ENV_ROOT/lib/python$PY_VERSION/site-packages"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
if [ -n "${LINUX_PROJECT:-}" ]; then
  export HF_HOME="${HF_HOME:-$LINUX_PROJECT/.cache/huggingface}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
fi

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

CUDA_HOME="${CUDA_HOME:-$SITE_PACKAGES/nvidia/cu13}"
if [ -d "$CUDA_HOME" ]; then
  export CUDA_HOME

  if [ -d "$CUDA_HOME/lib" ]; then
    ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
    if [ -f "$CUDA_HOME/lib/libcudart.so.13" ]; then
      ln -sfn "$CUDA_HOME/lib/libcudart.so.13" "$CUDA_HOME/lib/libcudart.so"
    fi
    mkdir -p "$CUDA_HOME/lib/stubs"
    if [ -f /usr/lib/wsl/lib/libcuda.so ]; then
      ln -sfn /usr/lib/wsl/lib/libcuda.so "$CUDA_HOME/lib/stubs/libcuda.so"
    fi
  fi

  NVIDIA_LIB_DIRS="$(find "$SITE_PACKAGES/nvidia" -mindepth 2 -maxdepth 3 -type d -name lib 2>/dev/null | paste -sd: -)"
  export LIBRARY_PATH="$CUDA_HOME/lib:$NVIDIA_LIB_DIRS:/usr/lib/wsl/lib:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="$CUDA_HOME/lib:$NVIDIA_LIB_DIRS:/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}"
fi

unset CCCL_DISABLE_CTK_COMPATIBILITY_CHECK
unset NVCC_PREPEND_FLAGS

if [ "${ENABLE_FLASHINFER_SAMPLER:-1}" = "1" ]; then
  unset VLLM_USE_FLASHINFER_SAMPLER
else
  export VLLM_USE_FLASHINFER_SAMPLER=0
fi

# Keep this Linux-only. WSL can inject Windows PATH entries containing spaces or
# parentheses, which are easy to leak into nested bash invocations.
export PATH="$PY_BIN:${CUDA_HOME:-}/bin:$HOME/.elan/bin:/opt/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/lib/wsl/lib"

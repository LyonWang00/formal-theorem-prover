#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/home/lean/miniforge3}"
ENV_NAME="${ENV_NAME:-lean_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  echo "missing conda at $CONDA_ROOT/bin/conda" >&2
  exit 1
fi

if "$CONDA_ROOT/bin/conda" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "$ENV_NAME exists"
else
  "$CONDA_ROOT/bin/mamba" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

"$CONDA_ROOT/envs/$ENV_NAME/bin/python" --version
"$CONDA_ROOT/envs/$ENV_NAME/bin/python" -m pip --version

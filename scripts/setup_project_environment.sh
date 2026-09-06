#!/usr/bin/env bash
set -euo pipefail

# One-command environment bootstrap for Linux/WSL users.
#
# Defaults match the development environment used by the Lean training and
# Pantograph benchmark pipeline. Override variables when needed, for example:
#
#   CONDA_ROOT=/path/to/conda ENV_NAME=lean_env bash scripts/setup_project_environment.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LEAN_PROVER_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
CONDA_ROOT="${CONDA_ROOT:-}"
ENV_NAME="${ENV_NAME:-lean_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
LEAN_PROJECT="${LEAN_PROJECT:-$ROOT/lean_project}"
BUNDLED_PANTOGRAPH_VERSION="0.3.15"
PANTOGRAPH_VERSION="${PANTOGRAPH_VERSION:-$BUNDLED_PANTOGRAPH_VERSION}"
PANTOGRAPH_SOURCE="${PANTOGRAPH_SOURCE:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_LAKE_CACHE="${RUN_LAKE_CACHE:-1}"

PROJECT_OWNER_UID="$(stat -c '%u' "$ROOT")"
if [ "${EUID:-$(id -u)}" = "0" ] && [ "$PROJECT_OWNER_UID" != "0" ]; then
  echo "Refusing to run as root against a project owned by UID $PROJECT_OWNER_UID." >&2
  echo "Run this script as the project owner to avoid root-owned Lake and model caches." >&2
  exit 2
fi

if [ -z "$CONDA_ROOT" ] && command -v conda >/dev/null 2>&1; then
  CONDA_ROOT="$(conda info --base)"
fi
if [ -z "$CONDA_ROOT" ]; then
  for conda_candidate in \
    "$HOME/miniforge3" \
    "$HOME/mambaforge" \
    "$HOME/miniconda3" \
    "/opt/conda"; do
    if [ -x "$conda_candidate/bin/conda" ]; then
      CONDA_ROOT="$conda_candidate"
      break
    fi
  done
fi
if [ ! -x "$CONDA_ROOT/bin/conda" ]; then
  echo "Unable to locate conda." >&2
  echo "Install Miniforge/Mambaforge/Miniconda, add conda to PATH, or set CONDA_ROOT." >&2
  exit 2
fi

CONDA="$CONDA_ROOT/bin/conda"
MAMBA="$CONDA_ROOT/bin/mamba"
if [ ! -x "$MAMBA" ]; then
  MAMBA="$CONDA"
fi

if "$CONDA" env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment already exists: $ENV_NAME"
else
  "$MAMBA" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi

PY="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys; print(sys.executable)')"
if [ ! -x "$PY" ]; then
  echo "Missing Python executable: $PY" >&2
  exit 2
fi

export PY
export LINUX_PROJECT="$ROOT"
export HF_ENDPOINT
source "$ROOT/scripts/lean_env_runtime.sh"
mkdir -p "$ROOT/outputs"

for required_command in git lake; do
  if ! command -v "$required_command" >/dev/null 2>&1; then
    echo "Missing required command: $required_command" >&2
    echo "Install Git and Elan/Lean before running this setup script." >&2
    exit 2
  fi
done

echo "==== Python ===="
"$PY" --version
"$PY" -m pip --version

echo
echo "==== Install Python dependencies ===="
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r "$ROOT/requirements_lean_env.txt"

echo
echo "==== Install Pantograph / PyPantograph $PANTOGRAPH_VERSION ===="
if [ -z "$PANTOGRAPH_SOURCE" ]; then
  if [ "$PANTOGRAPH_VERSION" = "$BUNDLED_PANTOGRAPH_VERSION" ] && \
    [ -f "$ROOT/.vendor/PyPantograph/pyproject.toml" ]; then
    PANTOGRAPH_SOURCE="$ROOT/.vendor/PyPantograph"
  else
    PANTOGRAPH_SOURCE="git+https://github.com/stanford-centaur/PyPantograph.git@v$PANTOGRAPH_VERSION"
  fi
fi
if [ -f "$PANTOGRAPH_SOURCE/pyproject.toml" ]; then
  "$PY" -m pip install --upgrade --no-deps "$PANTOGRAPH_SOURCE"
else
  "$PY" -m pip install --upgrade --no-deps "pantograph @ $PANTOGRAPH_SOURCE"
fi

INSTALLED_PANTOGRAPH_VERSION="$("$PY" -c 'from importlib.metadata import version; print(version("pantograph"))')"
if [ "$INSTALLED_PANTOGRAPH_VERSION" != "$PANTOGRAPH_VERSION" ]; then
  echo "Pantograph version mismatch: expected $PANTOGRAPH_VERSION, got $INSTALLED_PANTOGRAPH_VERSION" >&2
  exit 2
fi
echo "Pantograph source: $PANTOGRAPH_SOURCE"
echo "Pantograph version: $INSTALLED_PANTOGRAPH_VERSION"

# Refresh paths now that pip has populated CUDA, vLLM, and Pantograph packages
# in a newly created environment.
source "$ROOT/scripts/lean_env_runtime.sh"

echo
echo "==== Runtime paths ===="
echo "PY=$PY"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "nvcc=$(command -v nvcc || true)"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "LEAN_PROJECT=$LEAN_PROJECT"

if [ "$RUN_LAKE_CACHE" = "1" ]; then
  echo
  echo "==== Lean/mathlib cache ===="
  (cd "$LEAN_PROJECT" && lake --no-cache exe cache get)
fi

echo
echo "==== Direct Lean/mathlib smoke test ===="
(cd "$LEAN_PROJECT" && lake env lean "$ROOT/tests/lean_training_smoke.lean")

if [ "$RUN_PREFLIGHT" = "1" ]; then
  echo
  echo "==== Preflight ===="
  PYTHONPATH="$ROOT" "$PY" "$ROOT/scripts/preflight_lean_env.py" \
    --lean_project_path "$LEAN_PROJECT" \
    --timeout 1200 \
    --output_json "$ROOT/outputs/preflight_setup_project_environment.json"
fi

cat <<EOF

Environment setup completed.

Activate it with:
  source "$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  export PY="$(command -v python)"
  export LINUX_PROJECT="$ROOT"
  source "$ROOT/scripts/lean_env_runtime.sh"
EOF

#!/usr/bin/env bash
set -u

WINDOWS_ROOT="${WINDOWS_ROOT:-/mnt/e/python_project}"
LINUX_PROJECT="${LINUX_PROJECT:-/home/lean/projects/formal-theorem-prover}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-$WINDOWS_ROOT/outputs/linux_pantograph_$RUN_ID}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
HF_HOME="${HF_HOME:-$LINUX_PROJECT/.cache/huggingface}"

mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/run.log") 2>&1

step() {
  printf '\n==== %s ====\n' "$1"
}

run_logged() {
  local label="$1"
  shift
  step "$label"
  "$@"
}

require_path() {
  if [ ! -e "$1" ]; then
    echo "missing required path: $1"
    exit 2
  fi
}

step "environment"
date
whoami
uname -a
echo "HOME=$HOME"
echo "PATH=$PATH"
echo "WINDOWS_ROOT=$WINDOWS_ROOT"
echo "LINUX_PROJECT=$LINUX_PROJECT"
echo "LOG_DIR=$LOG_DIR"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "HF_HOME=$HF_HOME"
require_path "$WINDOWS_ROOT"
require_path "$LINUX_PROJECT"

cd "$LINUX_PROJECT" || exit 2

if [ -d "$LINUX_PROJECT/lean_project" ]; then
  LEAN_PROJECT="$LINUX_PROJECT/lean_project"
else
  LEAN_PROJECT="$LINUX_PROJECT"
fi
echo "LEAN_PROJECT=$LEAN_PROJECT"
export HF_ENDPOINT
export HF_HOME
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME"

step "sync lean_training and Pantograph adapter"
mkdir -p "$LINUX_PROJECT/lean_prover/lean_training"
if command -v rsync >/dev/null 2>&1; then
  rsync -rv --delete --exclude '__pycache__/' "$WINDOWS_ROOT/lean_prover/lean_training/" "$LINUX_PROJECT/lean_prover/lean_training/"
else
  find "$LINUX_PROJECT/lean_prover/lean_training" -name '__pycache__' -type d -prune -exec rm -rf {} +
  cp -av "$WINDOWS_ROOT/lean_prover/lean_training/." "$LINUX_PROJECT/lean_prover/lean_training/"
  find "$LINUX_PROJECT/lean_prover/lean_training" -name '__pycache__' -type d -prune -exec rm -rf {} +
fi
mkdir -p "$LINUX_PROJECT/lean_prover/backends"
if command -v rsync >/dev/null 2>&1; then
  rsync -rv --delete --exclude '__pycache__/' "$WINDOWS_ROOT/lean_prover/backends/" "$LINUX_PROJECT/lean_prover/backends/"
else
  cp -v "$WINDOWS_ROOT/lean_prover/backends/." "$LINUX_PROJECT/lean_prover/backends/"
fi
if [ -d "$WINDOWS_ROOT/data/minif2f" ]; then
  LINUX_MINIF2F_DIR="$LINUX_PROJECT/.cache/datasets/minif2f"
  mkdir -p "$LINUX_MINIF2F_DIR"
  if command -v rsync >/dev/null 2>&1; then
    rsync -rv --delete "$WINDOWS_ROOT/data/minif2f/" "$LINUX_MINIF2F_DIR/"
  else
    cp -v "$WINDOWS_ROOT/data/minif2f/." "$LINUX_MINIF2F_DIR/"
  fi
  echo "synced local miniF2F dataset to: $LINUX_MINIF2F_DIR"
else
  echo "local Windows miniF2F directory not found: $WINDOWS_ROOT/data/minif2f"
fi

step "git state"
git -C "$LINUX_PROJECT" status --short || true

step "Lean toolchain and project files"
export PATH="$HOME/.elan/bin:/opt/elan/bin:$PATH"
find "$LEAN_PROJECT" -maxdepth 2 -name lean-toolchain -print -exec cat {} \;
find "$LEAN_PROJECT" -maxdepth 2 \( -name lakefile.lean -o -name lakefile.toml -o -name lake-manifest.json \) -print
TOOLCHAIN="$(cat "$LEAN_PROJECT/lean-toolchain")"
echo "expected toolchain=$TOOLCHAIN"
command -v elan || true
command -v lean || true
command -v lake || true
if command -v elan >/dev/null 2>&1; then
  elan --version || true
  elan toolchain list || true
  if ! lean --version >/dev/null 2>&1; then
    echo "No active elan default toolchain; setting default to $TOOLCHAIN"
    elan default "$TOOLCHAIN" || {
      echo "elan default failed; trying elan toolchain install $TOOLCHAIN"
      elan toolchain install "$TOOLCHAIN" && elan default "$TOOLCHAIN"
    }
  fi
fi
lean --version || {
  echo "Lean is still unavailable after elan setup"
  exit 2
}
lake --version || {
  echo "Lake is still unavailable after elan setup"
  exit 2
}

step "Check and repair batteries dependency"
BATTERIES_DIR="$LEAN_PROJECT/.lake/packages/batteries"
if [ -d "$BATTERIES_DIR/.git" ]; then
  echo "batteries status before repair:"
  git -C "$BATTERIES_DIR" status --short || true
  if [ -n "$(git -C "$BATTERIES_DIR" status --short)" ]; then
    echo "batteries has local changes; showing summary before repair"
    git -C "$BATTERIES_DIR" diff --summary || true
    git -C "$BATTERIES_DIR" diff --stat || true
    git -C "$BATTERIES_DIR" restore .
    git -C "$BATTERIES_DIR" clean -fd
    echo "batteries status after repair:"
    git -C "$BATTERIES_DIR" status --short
  else
    echo "batteries is clean"
  fi
else
  echo "batteries dependency directory not found: $BATTERIES_DIR"
fi

step "Mathlib import smoke"
if [ -f "$LEAN_PROJECT/PantographImportSmoke.lean" ]; then
  timeout 900 lake -d "$LEAN_PROJECT" env lean "$LEAN_PROJECT/PantographImportSmoke.lean" || {
    echo "PantographImportSmoke failed or timed out"
  }
else
  SMOKE="$LOG_DIR/MathlibSmoke.lean"
  printf 'import Mathlib\n\n#check Nat\n' > "$SMOKE"
  timeout 900 lake -d "$LEAN_PROJECT" env lean "$SMOKE" || {
    echo "Mathlib smoke failed or timed out"
  }
fi

step "Python environment"
if [ -x "$LINUX_PROJECT/.venv/bin/python" ]; then
  PY="$LINUX_PROJECT/.venv/bin/python"
elif [ -x "/opt/venv/bin/python" ]; then
  PY="/opt/venv/bin/python"
elif [ -x "$HOME/.venv/bin/python" ]; then
  PY="$HOME/.venv/bin/python"
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
  PY="$VIRTUAL_ENV/bin/python"
elif command -v micromamba >/dev/null 2>&1; then
  PY="$(micromamba run -n leanprover_env which python 2>/dev/null || true)"
elif command -v conda >/dev/null 2>&1; then
  PY="$(conda run -n leanprover_env which python 2>/dev/null || true)"
elif command -v python >/dev/null 2>&1 && [ "$(command -v python)" != "/usr/bin/python" ]; then
  PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  echo "python3 not found"
  exit 2
fi
if [ -z "${PY:-}" ]; then
  echo "No Python interpreter found"
  exit 2
fi
echo "PY=$PY"
"$PY" --version
if ! "$PY" -m pip --version >/dev/null 2>&1; then
  echo "Selected Python has no pip: $PY"
  if [ "$PY" = "$(command -v python3 2>/dev/null || true)" ]; then
    echo "Trying to create project virtualenv at $LINUX_PROJECT/.venv"
    "$PY" -m venv "$LINUX_PROJECT/.venv" || {
      echo "python3 -m venv failed. Install python3-venv/python3-pip in WSL, then rerun."
      echo "Suggested PowerShell command:"
      echo "wsl -d Ubuntu-24.04 -u root -e bash -lc \"apt-get update && apt-get install -y python3-venv python3-pip\""
      exit 3
    }
    PY="$LINUX_PROJECT/.venv/bin/python"
  else
    echo "Install pip for this Python environment or provide $LINUX_PROJECT/.venv/bin/python"
    exit 3
  fi
fi
"$PY" -m pip --version

step "Python package check"
"$PY" - <<'PY' || MISSING_PACKAGES=1
mods = [
    "pantograph",
    "torch",
    "transformers",
    "peft",
    "datasets",
    "trl",
    "bitsandbytes",
    "accelerate",
]
missing = []
for mod in mods:
    try:
        __import__(mod)
        print(f"ok: {mod}")
    except Exception as exc:
        print(f"missing/broken: {mod}: {exc}")
        missing.append(mod)
if missing:
    raise SystemExit(1)
PY

if [ "${MISSING_PACKAGES:-0}" = "1" ]; then
  step "Install missing Python packages"
  if [[ "$PY" == /usr/bin/python* ]]; then
    echo "System Python is externally managed; creating project virtualenv at $LINUX_PROJECT/.venv"
    "$PY" -m venv "$LINUX_PROJECT/.venv" || {
      echo "python3 -m venv failed. Install python3-venv/python3-pip in WSL, then rerun."
      exit 3
    }
    PY="$LINUX_PROJECT/.venv/bin/python"
    "$PY" --version
    "$PY" -m pip --version
  fi
  "$PY" -m pip install --upgrade pip setuptools wheel
  if [ -d "$LINUX_PROJECT/.vendor/PyPantograph" ]; then
    "$PY" -m pip install -e "$LINUX_PROJECT/.vendor/PyPantograph"
  fi
  "$PY" -m pip install -e "$LINUX_PROJECT" \
    transformers peft datasets trl bitsandbytes accelerate safetensors || {
      echo "Python package installation failed"
      exit 3
    }
fi

step "Pantograph import/version check"
"$PY" - <<'PY'
from pantograph.server import get_version
print("Pantograph", get_version())
PY

step "Pantograph four-part benchmark"
timeout 3600 "$PY" -m lean_prover.lean_training.benchmark_pantograph \
  --lean_project_path "$LEAN_PROJECT" \
  --imports Mathlib \
  --timeout 180 \
  --warmup_timeout 1200 \
  --output_json "$LOG_DIR/pantograph_benchmark.json" || {
    echo "Pantograph benchmark failed or timed out"
    exit 4
  }

step "Prepare benchmark sample"
BENCHMARK_FILE="$LOG_DIR/benchmark4.jsonl"
LOCAL_MINIF2F="$LINUX_PROJECT/.cache/datasets/minif2f/test.jsonl"
if [ -f "$LOCAL_MINIF2F" ]; then
  echo "using local miniF2F dataset: $LOCAL_MINIF2F"
  "$PY" -m lean_prover.lean_training.prepare_datasets \
    --benchmark_dataset_name "$LOCAL_MINIF2F" \
    --benchmark_sample_size 4 \
    --benchmark_output "$BENCHMARK_FILE" \
    --seed 20260711 || {
      echo "prepare_datasets failed for local miniF2F file: $LOCAL_MINIF2F"
      exit 5
    }
else
  echo "local miniF2F dataset missing: $LOCAL_MINIF2F"
  "$PY" -m lean_prover.lean_training.prepare_datasets \
    --benchmark_dataset_name miniF2F \
    --benchmark_sample_size 4 \
    --benchmark_output "$BENCHMARK_FILE" \
    --seed 20260711 || {
      FALLBACK="$WINDOWS_ROOT/outputs/pipeline_tests/small_20260710_200724_fix/eval.jsonl"
      if [ -f "$FALLBACK" ]; then
        echo "prepare_datasets failed; using fallback benchmark file: $FALLBACK"
        cp "$FALLBACK" "$BENCHMARK_FILE"
      else
        echo "prepare_datasets failed and fallback benchmark file is unavailable"
        exit 5
      fi
    }
fi

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER="$WINDOWS_ROOT/outputs/runs/large_20260710_133248/adapter"

step "Small pipeline test: base model"
BASE_OUT="$LOG_DIR/base_pantograph"
timeout 5400 "$PY" -m lean_prover.lean_training.benchmark_pipeline \
  --model_name_or_path "$MODEL" \
  --benchmark_file "$BENCHMARK_FILE" \
  --output_dir "$BASE_OUT" \
  --num_benchmark_samples 4 \
  --num_workers 2 \
  --pass_k 2 \
  --generation_backend auto \
  --max_new_tokens 128 \
  --temperature 0.7 \
  --top_p 0.95 \
  --load_in_4bit \
  --lean_project_path "$LEAN_PROJECT" \
  --imports Mathlib \
  --warmup_timeout 1200 \
  --lean_timeout 180 || {
    echo "Base pipeline test failed or timed out"
    exit 6
  }

step "Small pipeline test: adapter model"
ADAPTER_OUT="$LOG_DIR/adapter_pantograph"
if [ "${RUN_ADAPTER_TEST:-0}" = "1" ] && [ -d "$ADAPTER" ]; then
  timeout 5400 "$PY" -m lean_prover.lean_training.benchmark_pipeline \
    --model_name_or_path "$MODEL" \
    --adapter_path "$ADAPTER" \
    --benchmark_file "$BENCHMARK_FILE" \
    --output_dir "$ADAPTER_OUT" \
    --num_benchmark_samples 4 \
    --num_workers 2 \
    --pass_k 2 \
    --generation_backend auto \
    --max_new_tokens 128 \
    --temperature 0.7 \
    --top_p 0.95 \
    --load_in_4bit \
    --lean_project_path "$LEAN_PROJECT" \
    --imports Mathlib \
    --warmup_timeout 1200 \
    --lean_timeout 180 || {
      echo "Adapter pipeline test failed or timed out"
      exit 7
    }
else
  echo "Adapter test skipped. Set RUN_ADAPTER_TEST=1 to enable it."
fi

step "Pipeline health summary"
"$PY" - <<PY
import json
from pathlib import Path
root = Path("$LOG_DIR")
for name in ("base_pantograph", "adapter_pantograph"):
    summary_path = root / name / "summary.json"
    attempts_path = root / name / "attempts.jsonl"
    if not summary_path.exists():
        print(f"{name}: summary missing")
        continue
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    attempts = []
    if attempts_path.exists():
        attempts = [
            json.loads(line)
            for line in attempts_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    timed_out = [item for item in attempts if item.get("timed_out")]
    rejected = [item for item in attempts if item.get("status") == "rejected"]
    success = [item for item in attempts if item.get("success")]
    failed = [item for item in attempts if item.get("status") == "failed"]
    abnormal = [
        item for item in attempts
        if item.get("status") not in {"success", "failed", "rejected", "canceled"}
    ]
    print(f"{name}:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"attempts={len(attempts)} success={len(success)} failed={len(failed)} rejected={len(rejected)} timed_out={len(timed_out)} abnormal={len(abnormal)}")
PY

step "done"
echo "Logs and results are in: $LOG_DIR"



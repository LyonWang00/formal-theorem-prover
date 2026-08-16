#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_sft_ddp.sh <1|2|4> train <existing SFT arguments...>
  scripts/run_sft_ddp.sh <1|2|4> smoke [smoke-test arguments...]

The launcher does not set learning rate, batch size, accumulation, epochs,
LoRA, data, evaluation, or checkpoint policy. Pass the same experiment
arguments used by the existing single-process command.
EOF
}

if [ "$#" -lt 2 ]; then
  usage >&2
  exit 2
fi

NUM_GPUS="$1"
MODE="$2"
shift 2
if [[ "$NUM_GPUS" != "1" && "$NUM_GPUS" != "2" && "$NUM_GPUS" != "4" ]]; then
  echo "num_gpus must be 1, 2, or 4" >&2
  exit 2
fi
if [[ "$MODE" != "train" && "$MODE" != "smoke" ]]; then
  echo "mode must be train or smoke" >&2
  usage >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-$PROJECT_ROOT/.venv/bin/accelerate}"
if [ ! -x "$PYTHON_BIN" ] || [ ! -x "$ACCELERATE_BIN" ]; then
  echo "missing project Python/Accelerate environment under $PROJECT_ROOT/.venv" >&2
  exit 2
fi

VISIBLE_GPU_COUNT="$($PYTHON_BIN -c 'import torch; print(torch.cuda.device_count())')"
if [ "$VISIBLE_GPU_COUNT" -lt "$NUM_GPUS" ]; then
  echo "requested $NUM_GPUS GPUs but only $VISIBLE_GPU_COUNT are visible" >&2
  exit 2
fi

LAUNCH_ARGS=(
  launch
  --num_machines 1
  --num_processes "$NUM_GPUS"
  --mixed_precision no
  --dynamo_backend no
)
if [ "$NUM_GPUS" -gt 1 ]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

cd "$PROJECT_ROOT"
if [ "$MODE" = "train" ]; then
  if [ "$#" -eq 0 ]; then
    echo "train mode requires the existing SFT arguments" >&2
    usage >&2
    exit 2
  fi
  exec "$ACCELERATE_BIN" "${LAUNCH_ARGS[@]}" \
    -m lean_prover.lean_training.sft --device_map local_rank "$@"
else
  exec "$ACCELERATE_BIN" "${LAUNCH_ARGS[@]}" \
    "$PROJECT_ROOT/scripts/sft_ddp_smoke.py" "$@"
fi

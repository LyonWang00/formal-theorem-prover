#!/usr/bin/env bash
set -euo pipefail

WINDOWS_ROOT="${WINDOWS_ROOT:-/mnt/e/python_project}"
LINUX_PROJECT="${LINUX_PROJECT:-/home/lean/projects/formal-theorem-prover}"
SAMPLE_SIZE="${SAMPLE_SIZE:-100}"
RUN_NAME="${RUN_NAME:-lean_workbook_sft_${SAMPLE_SIZE}}"
SEED="${SEED:-20260711}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_STEPS="${SAVE_STEPS:-200}"
VALIDATION_RATIO="${VALIDATION_RATIO:-0.02}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
ENABLE_FLASHINFER_SAMPLER="${ENABLE_FLASHINFER_SAMPLER:-1}"

PY="${PY:-$LINUX_PROJECT/.venv/bin/python}"
OUT_ROOT="$LINUX_PROJECT/outputs/${RUN_NAME}_${RUN_ID}"
DATA_FILE="$OUT_ROOT/train${SAMPLE_SIZE}.jsonl"
VALIDATION_FILE="$OUT_ROOT/validation${SAMPLE_SIZE}.jsonl"
MODEL_OUT="$OUT_ROOT/model"
LOG_FILE="$OUT_ROOT/train.log"

export HF_ENDPOINT
source "$LINUX_PROJECT/scripts/lean_env_runtime.sh"

mkdir -p "$OUT_ROOT" "$HF_HOME"
cd "$LINUX_PROJECT"

echo "WINDOWS_ROOT=$WINDOWS_ROOT"
echo "LINUX_PROJECT=$LINUX_PROJECT"
echo "PY=$PY"
echo "MODEL_NAME=$MODEL_NAME"
echo "SAMPLE_SIZE=$SAMPLE_SIZE"
echo "OUT_ROOT=$OUT_ROOT"
echo "DATA_FILE=$DATA_FILE"
echo "VALIDATION_FILE=$VALIDATION_FILE"
echo "MODEL_OUT=$MODEL_OUT"
echo "HF_ENDPOINT=$HF_ENDPOINT"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "PATH=$PATH"

if [ ! -x "$PY" ]; then
  echo "missing python environment: $PY"
  exit 2
fi

echo
echo "==== sync training code ===="
mkdir -p "$LINUX_PROJECT/lean_prover/lean_training"
if command -v rsync >/dev/null 2>&1; then
  rsync -rv --delete --exclude '__pycache__/' "$WINDOWS_ROOT/lean_prover/lean_training/" "$LINUX_PROJECT/lean_prover/lean_training/"
else
  cp -v "$WINDOWS_ROOT/lean_prover/lean_training/." "$LINUX_PROJECT/lean_prover/lean_training/"
fi
mkdir -p "$LINUX_PROJECT/lean_prover/backends"
if command -v rsync >/dev/null 2>&1; then
  rsync -rv --delete --exclude '__pycache__/' "$WINDOWS_ROOT/lean_prover/backends/" "$LINUX_PROJECT/lean_prover/backends/"
else
  cp -v "$WINDOWS_ROOT/lean_prover/backends/." "$LINUX_PROJECT/lean_prover/backends/"
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  echo
  echo "==== environment preflight ===="
  PREFLIGHT_ARGS=(
    "$LINUX_PROJECT/scripts/preflight_lean_env.py"
    --lean_project_path "$LINUX_PROJECT/lean_project"
    --timeout 1200
    --output_json "$OUT_ROOT/preflight.json"
  )
  if [ "$ENABLE_FLASHINFER_SAMPLER" != "1" ]; then
    PREFLIGHT_ARGS+=(--skip_flashinfer_smoke)
  fi
  "$PY" "${PREFLIGHT_ARGS[@]}"
fi

echo
echo "==== prepare Lean-Workbook train data ===="
echo "prepare_start=$(date -Is)"
/usr/bin/time -f "prepare_elapsed_seconds=%e" "$PY" -m lean_prover.lean_training.data.cli \
  --train_dataset_name lean-workbook \
  --train_sample_size "$SAMPLE_SIZE" \
  --train_output "$DATA_FILE" \
  --validation_output "$VALIDATION_FILE" \
  --validation_ratio "$VALIDATION_RATIO" \
  --tokenizer_name_or_path "$MODEL_NAME" \
  --max_length "$MAX_SEQ_LENGTH" \
  --seed "$SEED" 2>&1 | tee "$OUT_ROOT/prepare.log"
echo "prepare_end=$(date -Is)"

echo
echo "==== validate prepared train data ===="
"$PY" - "$DATA_FILE" "$VALIDATION_FILE" "$SAMPLE_SIZE" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
validation_path = Path(sys.argv[2])
expected_total = int(sys.argv[3])

def read_jsonl(item):
    return [
        json.loads(line)
        for line in item.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

rows = read_jsonl(path)
validation_rows = read_jsonl(validation_path) if validation_path.exists() else []
all_rows = rows + validation_rows
bad = [
    row.get("id")
    for row in all_rows
    if re.search(r"\b(?:sorry|admit)\b", row.get("proof", ""))
]
missing_text = [row.get("id") for row in all_rows if not row.get("text")]
missing_proof = [row.get("id") for row in all_rows if not row.get("proof")]
print(f"prepared_rows={len(rows)}")
print(f"prepared_validation_rows={len(validation_rows)}")
print(f"prepared_total_rows={len(all_rows)}")
print(f"forbidden_rows={len(bad)}")
print(f"missing_text_rows={len(missing_text)}")
print(f"missing_proof_rows={len(missing_proof)}")
if rows:
    first = rows[0]
    print("first_id=" + str(first.get("id")))
    print(
        "first_statement_escape="
        + first.get("lean_statement", "").encode("unicode_escape").decode("ascii")[:600]
    )
    print(
        "first_proof_escape="
        + first.get("proof", "").encode("unicode_escape").decode("ascii")[:600]
    )
if len(all_rows) != expected_total or bad or missing_text or missing_proof:
    raise SystemExit(1)
PY

echo
echo "==== train QLoRA adapter ===="
echo "train_start=$(date -Is)"
/usr/bin/time -f "train_elapsed_seconds=%e" "$PY" -m lean_prover.lean_training.sft \
  --model_name_or_path "$MODEL_NAME" \
  --train_file "$DATA_FILE" \
  --validation_file "$VALIDATION_FILE" \
  --output_dir "$MODEL_OUT" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" 2>&1 | tee "$LOG_FILE"
echo "train_end=$(date -Is)"

echo
echo "==== saved files ===="
find "$OUT_ROOT" -maxdepth 3 -type f | sort




#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/lean/projects/formal-theorem-prover
QUEUE="$PROJECT/outputs/numinamath_expand_verification/worker_0/queue"
LOG="$PROJECT/outputs/numinamath_expand_verification/worker_0/worker0_resume_lean.log"
MATHLIB_OLEAN="$PROJECT/lean_project/.lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"

if [[ "$(id -un)" != "lean" ]]; then
  echo "refusing to start worker0 as $(id -un); use sudo -u lean -H" >&2
  exit 64
fi
if [[ ! -f "$MATHLIB_OLEAN" ]]; then
  echo "refusing to start worker0 before Mathlib.olean is complete: $MATHLIB_OLEAN" >&2
  exit 65
fi
if [[ -e "$QUEUE/STOP" ]]; then
  echo "refusing to start worker0 while queue STOP exists: $QUEUE/STOP" >&2
  exit 66
fi

cd "$PROJECT"
exec env PYTHONPATH=. .venv/bin/python -m lean_prover.Dataset.numinamath_pantograph_worker \
  run \
  --queue-dir "$QUEUE" \
  --lean-project lean_project \
  --timeout 30 \
  >> "$LOG" 2>&1

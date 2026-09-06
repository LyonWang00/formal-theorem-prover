#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:?cloud run root required}
MODEL_LABEL=${2:?model label required}
RECEIPT_ORIGIN=${3:?receipt origin required}
cp "$ROOT/code/pool_hard_timeout.py" "$ROOT/code/lean_prover/lean_training/verification/pool.py"
test -f "$ROOT/COMPUTE_GENERATION_COMPLETE.json"
test ! -e "$ROOT/EVALUATION_COMPLETE.json"
setsid bash "$ROOT/code/cloud_compile.sh" "$ROOT" "$MODEL_LABEL" "$RECEIPT_ORIGIN" > "$ROOT/driver.log" 2>&1 < /dev/null &
echo $! > "$ROOT/driver.pid"
cat "$ROOT/driver.pid"

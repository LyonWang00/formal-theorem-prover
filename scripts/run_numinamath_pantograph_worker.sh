#!/usr/bin/env bash
set -euo pipefail

cd /home/lean/projects/formal-theorem-prover
exec env PYTHONPATH=. .venv/bin/python -m lean_prover.Dataset.numinamath_pantograph_worker \
  run \
  --queue-dir outputs/numinamath_repair/worker_queue \
  --lean-project lean_project \
  --timeout 30

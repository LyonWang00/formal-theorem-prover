#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f /workspace/pyproject.toml ]]; then
  echo "error: /workspace is not the project root" >&2
  exit 2
fi
if [[ ! -f "${LEAN_PROJECT_PATH}/lean-toolchain" ]]; then
  echo "error: Lean runtime project is missing: ${LEAN_PROJECT_PATH}" >&2
  exit 2
fi

exec "$@"

#!/usr/bin/env bash
set -euo pipefail

project="${LEAN_PROJECT_PATH:-/opt/lean-project}"
expected="$(tr -d '[:space:]' < "$project/lean-toolchain")"
active="$(cd "$project" && elan show | awk 'NR == 1 {print $1}')"

if [[ "$active" != "$expected" ]]; then
  echo "toolchain mismatch: expected $expected, active $active" >&2
  exit 1
fi

cd "$project"
lean --version
lake --version
lake env lean --version

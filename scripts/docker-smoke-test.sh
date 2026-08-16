#!/usr/bin/env bash
set -euo pipefail

cd /workspace
project="${LEAN_PROJECT_PATH:-/opt/lean-project}"

echo "=== Environment ==="
echo "workspace=$(pwd)"
echo "lean_project=$project"
python --version
uv --version
elan --version
./scripts/check-toolchain.sh
python -c \
  "from pantograph.server import get_version; print('Pantograph', get_version())"
python -c \
  "import lean_prover; print('lean_prover import: OK')"

mathlib_commit="$(
  python -c \
    "import json; p=json.load(open('$project/lake-manifest.json')); print(next(x['rev'] for x in p['packages'] if x['name']=='mathlib'))"
)"
echo "Mathlib commit: $mathlib_commit"
test "$mathlib_commit" = \
  "5e932f97dd25535344f80f9dd8da3aab83df0fe6"

echo "=== Mathlib import ==="
cd "$project"
lake env lean PantographImportSmoke.lean

echo "=== Python unit tests ==="
cd /workspace
python -m pytest -q

echo "=== Pantograph acceptance test ==="
python -m scripts.pantograph_smoke_test

echo "ALL DOCKER SMOKE TESTS PASSED"

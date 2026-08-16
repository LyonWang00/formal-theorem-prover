#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LEAN_PROVER_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
SOURCE="$ROOT/lean_project"

test -f "$SOURCE/lean-toolchain"
test -f "$SOURCE/lakefile.lean" || test -f "$SOURCE/lakefile.toml"

SOURCE="$(realpath "$SOURCE")"
if [[ -n "${PANTOGRAPH_PROJECT_PATH:-}" ]]; then
  RUNTIME="$(realpath -m "$PANTOGRAPH_PROJECT_PATH")"
elif [[ "$SOURCE" == /mnt/?/* ]]; then
  # Windows-mounted filesystems are slow for Mathlib's many small artifacts.
  RUNTIME="$HOME/.local/share/lean-math-prover/lean_project"
else
  # Native Linux/WSL filesystems need no duplicate runtime tree.
  RUNTIME="$SOURCE"
fi

if [[ "$SOURCE" != "$RUNTIME" ]]; then
  mkdir -p "$RUNTIME"
  rsync -a --delete --exclude=.git --exclude=.lake "$SOURCE/" "$RUNTIME/"
fi

export PATH="$ROOT/scripts/wsl-bin:$PATH"
cd "$RUNTIME"
lake update
lake exe cache get
lake build

printf 'Pantograph runtime project: %s\n' "$RUNTIME"
printf 'Use: export LEAN_PROJECT_PATH=%q\n' "$RUNTIME"

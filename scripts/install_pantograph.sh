#!/usr/bin/env bash
set -euo pipefail

WRAPPER_COMMIT="ffa7f243824d2762825abddb1e9f6e939ede761f"
PANTOGRAPH_COMMIT="842c0fe6e76b0771cc7f7939604c7a0b90e17433"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LEAN_PROVER_ROOT:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
VENDOR="$ROOT/.vendor/PyPantograph"
PROXY="https://ghfast.top/https://github.com"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/lean-math-prover}"

mkdir -p "$ROOT/.vendor"
if [[ ! -d "$VENDOR/.git" ]]; then
  git clone "$PROXY/stanford-centaur/PyPantograph.git" "$VENDOR"
fi
git -C "$VENDOR" fetch "$PROXY/stanford-centaur/PyPantograph.git" \
  "$WRAPPER_COMMIT"
git -C "$VENDOR" checkout --detach "$WRAPPER_COMMIT"

if [[ ! -d "$VENDOR/src/.git" ]]; then
  rm -rf "$VENDOR/src"
  git clone "$PROXY/leanprover/Pantograph.git" "$VENDOR/src"
fi
git -C "$VENDOR/src" fetch "$PROXY/leanprover/Pantograph.git" \
  "$PANTOGRAPH_COMMIT"
git -C "$VENDOR/src" checkout --detach "$PANTOGRAPH_COMMIT"

actual_wrapper="$(git -C "$VENDOR" rev-parse HEAD)"
actual_pantograph="$(git -C "$VENDOR/src" rev-parse HEAD)"
[[ "$actual_wrapper" == "$WRAPPER_COMMIT" ]]
[[ "$actual_pantograph" == "$PANTOGRAPH_COMMIT" ]]
gitlink="$(git -C "$VENDOR" ls-tree HEAD src | awk '{print $3}')"
[[ "$gitlink" == "$PANTOGRAPH_COMMIT" ]]
grep -qx 'leanprover/lean4:v4.29.1' "$VENDOR/src/lean-toolchain"

cd "$ROOT"
# Apply the accelerator only to Git subprocesses spawned by this installer.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0="url.${PROXY}/.insteadOf"
export GIT_CONFIG_VALUE_0="https://github.com/"
uv sync --extra dev --no-install-project
"$UV_PROJECT_ENVIRONMENT/bin/python" -c \
  'import pantograph; print("PyPantograph installation complete")'

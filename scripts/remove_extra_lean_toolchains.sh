#!/usr/bin/env bash
set -euo pipefail

KEEP="${KEEP:-leanprover/lean4:v4.29.1}"
REMOVE_TOOLCHAINS=(
  "leanprover/lean4:v4.30.0"
  "leanprover/lean4:v4.30.0-rc2"
)

echo "Current default/toolchains:"
elan show || true
elan toolchain list || true

echo
echo "Keeping required toolchain: $KEEP"
elan toolchain install "$KEEP"
elan default "$KEEP"

for toolchain in "${REMOVE_TOOLCHAINS[@]}"; do
  echo
  echo "Removing unused toolchain: $toolchain"
  elan toolchain uninstall "$toolchain" || true
done

echo
echo "Remaining toolchains:"
elan toolchain list
lean --version
lake --version

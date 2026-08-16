#!/usr/bin/env bash
set -euo pipefail

cd /home/lean/projects/formal-theorem-prover
root="outputs/initial_anchor_ratio_ablation"
mkdir -p "${root}/runtime/logs"

while pgrep -f "scripts.train_initial_anchor_ratio_arm --arm A0_WB3000_LD0" >/dev/null; do
  sleep 30
done
test -f "${root}/training/A0_WB3000_LD0/training_summary.json"

for arm in A5_WB2750_LD250 A10_WB2500_LD500 A20_WB2000_LD1000; do
  if test -f "${root}/training/${arm}/training_summary.json"; then
    continue
  fi
  PYTHONPATH=. .venv/bin/python -m scripts.train_initial_anchor_ratio_arm \
    --arm "${arm}" >"${root}/runtime/logs/train_${arm}.log" 2>&1
done

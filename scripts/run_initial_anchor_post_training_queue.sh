#!/usr/bin/env bash
set -euo pipefail

cd /home/lean/projects/formal-theorem-prover
root="outputs/initial_anchor_ratio_ablation"
mkdir -p "${root}/runtime/logs"

for arm in A0_WB3000_LD0 A5_WB2750_LD250 A10_WB2500_LD500 A20_WB2000_LD1000; do
  while ! test -f "${root}/training/${arm}/training_summary.json"; do
    sleep 30
  done
done

if ! test -f "${root}/final_report.md"; then
  PYTHONPATH=. .venv/bin/python -m scripts.run_initial_anchor_ratio_evaluation \
    >"${root}/runtime/logs/evaluation_queue.log" 2>&1
fi

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/ld_length_difficulty_pipeline/length_ablation"

cd "${PROJECT_ROOT}"
exec "${PROJECT_ROOT}/.venv/bin/python" \
  scripts/orchestrate_ld_length_ablation.py \
  >> "${OUTPUT_ROOT}/runtime/background_orchestrator.log" 2>&1

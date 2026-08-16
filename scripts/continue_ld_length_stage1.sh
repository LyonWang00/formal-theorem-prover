#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/ld_length_difficulty_pipeline/length_ablation"
PROGRESS="${OUTPUT_ROOT}/runtime/orchestrator_progress.json"
CHAIN_STATUS="${OUTPUT_ROOT}/runtime/stage1_chain_status.json"
CHAIN_LOG="${OUTPUT_ROOT}/runtime/stage1_chain.log"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/runtime"
printf '{"status":"waiting_for_canary"}\n' > "${CHAIN_STATUS}"

while true; do
  status="$(
    "${PROJECT_ROOT}/.venv/bin/python" -c \
      "import json; print(json.load(open('${PROGRESS}', encoding='utf-8')).get('status','missing'))" \
      2>/dev/null || printf 'missing'
  )"
  case "${status}" in
    completed)
      break
      ;;
    failed|interrupted)
      printf '{"status":"blocked","canary_status":"%s"}\n' "${status}" > "${CHAIN_STATUS}"
      exit 1
      ;;
    *)
      sleep 30
      ;;
  esac
done

printf '{"status":"analyzing_canary"}\n' > "${CHAIN_STATUS}"
"${PROJECT_ROOT}/.venv/bin/python" scripts/analyze_ld_length_ablation.py \
  >> "${CHAIN_LOG}" 2>&1
"${PROJECT_ROOT}/.venv/bin/python" scripts/report_ld_length_ablation.py \
  >> "${CHAIN_LOG}" 2>&1

printf '{"status":"preparing_gated_expansion"}\n' > "${CHAIN_STATUS}"
"${PROJECT_ROOT}/.venv/bin/python" scripts/prepare_ld_length_expansion.py \
  >> "${CHAIN_LOG}" 2>&1

PLAN="${OUTPUT_ROOT}/manifests/expansion_plan.json"
if [[ -f "${PLAN}" ]]; then
  authorized="$(
    "${PROJECT_ROOT}/.venv/bin/python" -c \
      "import json; print(str(bool(json.load(open('${PLAN}', encoding='utf-8')).get('authorized'))).lower())"
  )"
else
  authorized=false
fi

if [[ "${authorized}" == "true" ]]; then
  printf '{"status":"running_gated_expansion"}\n' > "${CHAIN_STATUS}"
  "${PROJECT_ROOT}/.venv/bin/python" scripts/orchestrate_ld_length_expansion.py \
    >> "${CHAIN_LOG}" 2>&1
fi

printf '{"status":"stage1_runs_complete","expansion_authorized":%s}\n' \
  "${authorized}" > "${CHAIN_STATUS}"

#!/usr/bin/env bash
set -euo pipefail

ORCH=/home/lean/experiments/grpo-round67-orchestration-20260904
REMOTE_ROOT=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1
REMOTE_EVAL=$REMOTE_ROOT/eval_minif2f_step108_compute_generation_cloud_compile
POOL_SOURCE_ROOT=/home/scc/bz22001004/runs/grpo-round6-old-sft-numiter2-lr1e5-20260904-v1
CLOUD_ROOT=/home/lean/experiments/grpo-dynamic-step108-eval-20260906
SOCKET=/home/lean/.ssh/cm/ustc-scc
CLOUD=(ssh -i /home/lean/.ssh/eastwind_lean_ed25519 -p 1132 -o BatchMode=yes -o ServerAliveInterval=60 -o ServerAliveCountMax=10 lean@114.214.241.41)

ssh -S "$SOCKET" -o BatchMode=yes ustc-scc \
  "test -f '$REMOTE_EVAL/COMPUTE_GENERATION_COMPLETE.json'"
"${CLOUD[@]}" "test ! -e '$CLOUD_ROOT/EVALUATION_COMPLETE.json'"

ssh -S "$SOCKET" -o BatchMode=yes ustc-scc \
  "tar -C '$REMOTE_EVAL' -cf - generation compile_manifest_all32 audit/GENERATION_AUDIT.json COMPUTE_GENERATION_COMPLETE.json -C '$POOL_SOURCE_ROOT' code/pool_hard_timeout.py" \
  | "${CLOUD[@]}" bash /home/lean/cloud_receive_grpo_round67.sh "$CLOUD_ROOT"

tar -C "$ORCH" -cf - \
  code/grpo_two_phase.py code/minif2f_full_split_verify_nowarm.py \
  code/analyze_corrected.py code/lean_prover code/cloud_compile.sh \
  code/cloud_start_compile.sh input/frozen4 \
  | "${CLOUD[@]}" bash /home/lean/cloud_receive_grpo_round67.sh "$CLOUD_ROOT"

"${CLOUD[@]}" bash "$CLOUD_ROOT/code/cloud_start_compile.sh" \
  "$CLOUD_ROOT" grpo_dynamic_step108 cloud_grpo_dynamic_step108_minif2f

"${CLOUD[@]}" "test -s '$CLOUD_ROOT/driver.pid' && cat '$CLOUD_ROOT/driver.pid'"

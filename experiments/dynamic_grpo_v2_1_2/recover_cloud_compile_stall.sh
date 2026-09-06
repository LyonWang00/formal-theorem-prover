#!/usr/bin/env bash
set -euo pipefail

R=/home/lean/experiments/grpo-e2-adaptive-step149-pass32-20260905-cloud
O="$R/cloud_compile"
DRIVER=193584
VERIFY=193622

[[ $(wc -l < "$O/receipts.jsonl") -eq 7803 ]]
now=$(date +%s)
mtime=$(stat -c %Y "$O/receipts.jsonl")
(( now - mtime > 120 ))

kill -TERM "$VERIFY" 2>/dev/null || true
for _ in {1..10}; do
  kill -0 "$VERIFY" 2>/dev/null || break
  sleep 1
done
kill -TERM "$DRIVER" 2>/dev/null || true
sleep 2

# Workers start their own sessions; terminate only descendants from this
# verifier instance if graceful teardown did not already remove them.
for pid in 193640 193658; do
  kill -TERM "$pid" 2>/dev/null || true
done
sleep 2
for pid in "$VERIFY" "$DRIVER" 193640 193658; do
  kill -KILL "$pid" 2>/dev/null || true
done

ts=$(date -u +%Y%m%dT%H%M%SZ)
cp -- "$O/progress.json" "$O/progress.stalled-$ts.json"
cp -- "$O/verify.log" "$O/verify.stalled-$ts.log"

nohup env QUEUE_MAXSIZE=1 "$R/code/cloud_compile_resume1.sh" \
  "$R" grpo_adaptive_e2_step149 cloud_grpo_adaptive_e2_step149_minif2f \
  > "$R/audit/cloud_compile_resume-$ts.out" 2>&1 < /dev/null &
pid=$!
printf '{"action":"resume_after_stall","prior_driver_pid":%d,"prior_verifier_pid":%d,"receipt_count":7803,"new_driver_pid":%d,"timestamp":"%s"}\n' \
  "$DRIVER" "$VERIFY" "$pid" "$ts" | tee "$O/RECOVERY_$ts.json"

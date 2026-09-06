#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?cloud run root required}
MODEL_LABEL=${2:?model label required}
RECEIPT_ORIGIN=${3:?receipt origin required}
PROJECT=/home/lean/projects/formal-theorem-prover
ENV=$PROJECT/.venv
CODE=$ROOT/code
FROZEN=$ROOT/input/frozen4/SHARDS_FROZEN.json
GEN=$ROOT/generation
MANIFEST=$ROOT/compile_manifest_all32/compile_manifest.jsonl
OUTPUT=$ROOT/cloud_compile
SPOOL=/var/tmp/$USER-grpo-round67-cloud-$$
LOCK=/home/lean/experiments/formal-theorem-prover/.grpo-round67-pass32-cloud.lock

cleanup() {
  case "$(readlink -f "$SPOOL")" in
    (/var/tmp/"$USER"-grpo-round67-cloud-*) rm -rf -- "$SPOOL" ;;
    (*) return 1 ;;
  esac
}
trap cleanup EXIT INT TERM

test ! -e "$ROOT/EVALUATION_COMPLETE.json"
test "$(sha256sum "$CODE/grpo_two_phase.py" | awk '{print $1}')" = a7247e82f37f0e63708e7df0f4341849140db73c29453d6d8a11253bd2d6f354
test "$(sha256sum "$CODE/minif2f_full_split_verify_nowarm.py" | awk '{print $1}')" = 811e67b39b3726a4808b9e189e39fde0a80ed14da983cbb3990b198645a94d49
test "$(sha256sum "$CODE/lean_prover/lean_training/verification/pool.py" | awk '{print $1}')" = 6d50f0728574e3cb89b372dfa059b310bc435843d7c0c4680f300c37955dd10b
test "$(lean --version)" = "Lean (version 4.29.1, x86_64-unknown-linux-gnu, commit f72c35b3f637c8c6571d353742168ab66cc22c00, Release)"
test "$(git -C "$PROJECT/lean_project/.lake/packages/mathlib" rev-parse HEAD)" = 5e932f97dd25535344f80f9dd8da3aab83df0fe6
test "$(find "$PROJECT/lean_project/.lake" -type f -name '*.olean' | wc -l)" -ge 8241

readarray -t VALUES < <("$ENV/bin/python" - "$ROOT/COMPUTE_GENERATION_COMPLETE.json" <<'PY'
import json, pathlib, sys
p=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert p['status']=='PASS' and p['problem_count']==244 and p['attempt_count']==7808
print(p['model_sums_sha256'])
print(p['frozen_shards_sha256'])
print(p['compile_manifest_sha256'])
PY
)
MODEL_SUMS_SHA=${VALUES[0]}
FROZEN_SHA=${VALUES[1]}
MANIFEST_SHA=${VALUES[2]}
test "$(sha256sum "$FROZEN" | awk '{print $1}')" = "$FROZEN_SHA"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = "$MANIFEST_SHA"

# The two rounds share one cloud verifier.  Blocking on the lock makes round 7
# queue safely behind round 6 if its generation finishes unusually early.
exec 9>"$LOCK"
flock 9
test ! -e "$SPOOL"
install -d -m 700 "$SPOOL"
mkdir -p "$OUTPUT"

export PYTHONPATH=$CODE PYTHONUNBUFFERED=1
export LEAN_RECEIPT_ORIGIN=$RECEIPT_ORIGIN
export LEAN_EXECUTION_DOMAIN=cloud_pantograph

"$ENV/bin/python" -P "$CODE/grpo_two_phase.py" audit-generations \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GEN" --model-sums-sha256 "$MODEL_SUMS_SHA" \
  --output "$ROOT/CLOUD_GENERATION_AUDIT.json"

"$ENV/bin/python" -P "$CODE/minif2f_full_split_verify_nowarm.py" verify \
  --grpo-entrypoint "$CODE/grpo_two_phase.py" \
  --grpo-entrypoint-sha256 a7247e82f37f0e63708e7df0f4341849140db73c29453d6d8a11253bd2d6f354 \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GEN" --model-sums-sha256 "$MODEL_SUMS_SHA" \
  --compile-manifest "$MANIFEST" --compile-manifest-sha256 "$MANIFEST_SHA" \
  --receipt-origin "$RECEIPT_ORIGIN" --execution-domain cloud_pantograph \
  --output "$OUTPUT" --lean-project "$PROJECT/lean_project" --task-spool-dir "$SPOOL" \
  --workers 2 --timeout 30 --warmup-timeout 3600 --queue-maxsize 256 --serial-prewarm \
  > "$OUTPUT/verify.log" 2>&1

"$ENV/bin/python" -P - "$ROOT" "$MODEL_LABEL" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); label=sys.argv[2]
sys.path.insert(0, str(root/'code'))
from analyze_corrected import load_generations, load_grpo_receipts, summarize_model
summary=json.loads((root/'cloud_compile/summary.json').read_text())
assert summary['status']=='complete' and summary['receipt_count']==7808
assert summary['selected_attempt_count']==7808
assert summary['workers']==2 and summary['worker_lifecycle']=='persistent' and summary['prewarm_mode']=='serial'
assert not summary['fatal_errors']
generations=load_generations(root/'generation')
receipts=load_grpo_receipts(root/'cloud_compile/receipts.jsonl')
result=summarize_model(label, generations, receipts)
q=result['generation_quality']; c=result['compile_quality']
final={
  'schema':label+'_minif2f_test_pass32_cloud_result_v1',
  'status':'PASS','problem_count':244,'attempt_count':7808,
  'pass_at_k_actual_prefix':result['pass_at_k_actual_prefix'],
  'pass_at_k_unbiased_estimator':result['pass_at_k_unbiased_estimator'],
  'attempt_success_rate':result['attempt_success_rate'],
  'length_truncation_count':q['hit_max_new_tokens_count'],
  'length_truncation_rate':q['hit_max_new_tokens_count']/7808,
  'strict_hallucinated_api_attempt_count':c['strict_hallucinated_api_attempt_count'],
  'strict_hallucinated_api_attempt_rate':c['strict_hallucinated_api_attempt_rate'],
  'compile_workers':2,'worker_lifecycle':'persistent','serial_prewarm':True,
  'compile_timeout_seconds':30,'generation_host':'ustc_compute_rtx5090x4',
  'verification_host':'cloud_pantograph',
}
(root/'FINAL_RESULT.json').write_text(json.dumps(final,indent=2,sort_keys=True)+'\n')
(root/'EVALUATION_COMPLETE.json').write_text(json.dumps({'status':'PASS','final_result':final},indent=2,sort_keys=True)+'\n')
print(json.dumps(final,indent=2,sort_keys=True))
PY

#!/usr/bin/env bash
#SBATCH --job-name=grpo-m2f-p32
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append
#SBATCH --output=/home/scc/bz22001004/runs/minif2f-grpo-pass32-20260829-v1/logs/full-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/minif2f-grpo-pass32-20260829-v1/logs/full-%j.err

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/minif2f-grpo-pass32-20260829-v1
PROJECT=/home/scc/bz22001004/projects/formal-theorem-prover
OLD=/home/scc/bz22001004/runs/minif2f-resume18688-pass32-20260827
GRPO_RUN=/home/scc/bz22001004/runs/grpo-first5000-epoch2-20260829-v1
SFT_ENV=/home/scc/bz22001004/environments/formal-theorem-prover-sft
VLLM_ENV=/home/scc/bz22001004/environments/rollout-vllm-cu130
BASE="$PROJECT/models/DeepSeek-Prover-V1.5-Base"
ADAPTER="$GRPO_RUN/full/model"
TOKENIZER=/home/scc/bz22001004/runs/rollout-expand-20260824/contracts/deepseek-prover-v15-lean-whitespace-tokenizer-compat-v1
BNB_ROOT=/home/scc/bz22001004/runs/rollout-expand-20260824/vllm-bitsandbytes-0.49.2
BNB_MARKER="$BNB_ROOT/INSTALL_COMPLETE.json"
DATA="$OLD/input/full_test244_data.jsonl"
MANIFEST="$OLD/input/full_test244_manifest.jsonl"
AUDIT="$OLD/input/full_test244_audit.json"
NODE_ROOT=/var/tmp/$USER-minif2f-grpo-pass32-$SLURM_JOB_ID
NODE_CODE="$NODE_ROOT/code"
NODE_MODEL="$NODE_ROOT/merged-grpo-final"
FROZEN="$ROOT/frozen4/SHARDS_FROZEN.json"
GENERATION="$ROOT/generation"
COMPILE_MANIFEST="$ROOT/compile_manifest_all32"
COMPILE_OUTPUT="$ROOT/compile"
SPOOL="$NODE_ROOT/spool"

mkdir -p "$ROOT/logs" "$ROOT/audit" "$GENERATION" "$COMPILE_OUTPUT"
test ! -e "$NODE_ROOT"
mkdir -p "$NODE_CODE" "$SPOOL"

safe_cleanup() {
  case "$(readlink -f "$NODE_ROOT")" in
    (/var/tmp/"$USER"-minif2f-grpo-pass32-"$SLURM_JOB_ID") rm -rf -- "$NODE_ROOT" ;;
    (*) echo "unsafe node root; refusing cleanup" >&2 ;;
  esac
}
gpu_sampler_pid=""
cleanup() {
  set +e
  if [[ -n "$gpu_sampler_pid" ]]; then kill "$gpu_sampler_pid" 2>/dev/null || true; fi
  safe_cleanup
}
trap cleanup EXIT INT TERM USR1

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=/home/scc/bz22001004/.elan/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=4

tar -C "$PROJECT" --exclude='lean_prover/Dataset' --exclude='__pycache__' --exclude='*.pyc' -cf - lean_prover | tar -C "$NODE_CODE" -xf -
tar -C "$NODE_CODE" -xf "$OLD/staging/lean_training_runtime.tar"
cp "$OLD/staging/grpo_two_phase.py" "$NODE_CODE/grpo_two_phase.py"
cp "$OLD/staging/grpo_vllm_env_wrapper.py" "$NODE_CODE/grpo_vllm_env_wrapper.py"
cp "$ROOT/scripts/minif2f_full_split_verify_nowarm.py" "$NODE_CODE/minif2f_split_verify.py"
cp "$ROOT/scripts/merge_grpo_adapter.py" "$NODE_CODE/merge_grpo_adapter.py"
export PYTHONPATH="$NODE_CODE"

python - <<'PY' "$DATA"
import json,sys
rows=[json.loads(line) for line in open(sys.argv[1], encoding='utf-8') if line.strip()]
assert len(rows)==244, len(rows)
assert {row.get('split') for row in rows}=={'test'}
assert len({row['id'] for row in rows})==244
assert not any(any(key in row for key in ('proof','completion','reference_proof')) for row in rows)
print('TEST244_GATE PASS rows=244 split=test unique_ids=244 proof_fields=0')
PY

DATA_SHA=$(sha256sum "$DATA" | awk '{print $1}')
MANIFEST_SHA=$(sha256sum "$MANIFEST" | awk '{print $1}')
AUDIT_SHA=$(sha256sum "$AUDIT" | awk '{print $1}')
TOKENIZER_SUMS_SHA=$(sha256sum "$TOKENIZER/TOKENIZER_SHA256SUMS" | awk '{print $1}')
test "$DATA_SHA" = 2a261e530c74c619507d4efd1b4e42cf7f1c7a19c8458880f223d61b41a02d24
test "$MANIFEST_SHA" = 090f38e0f0be9d6439f7e6afef4ad154bb82651be60c8ae1f35392c39c945c1a
test "$AUDIT_SHA" = 8bef03cfb6b4f3ce811a8115e3ee499959aeb4c4614bc2e3019004d8acaa8389
test "$TOKENIZER_SUMS_SHA" = 197a57bff74df59755134613ba1940a085c33412d63739750204842c60eedd02

if [[ ! -f "$FROZEN" ]]; then
  "$VLLM_ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" freeze-shards \
    --data "$DATA" --data-sha256 "$DATA_SHA" \
    --manifest "$MANIFEST" --manifest-sha256 "$MANIFEST_SHA" \
    --audit "$AUDIT" --audit-sha256 "$AUDIT_SHA" \
    --output "$ROOT/frozen4" --shards 4 --pass-k 32 --expected-count 244 \
    --tokenizer-path "$TOKENIZER" --tokenizer-sums-sha256 "$TOKENIZER_SUMS_SHA" \
    --max-new-tokens 2048 --max-model-len 4096 --generation-batch-size 2 \
    --seed 20260711 --temperature 1.0 --top-p 0.95
fi
FROZEN_SHA=$(sha256sum "$FROZEN" | awk '{print $1}')

"$SFT_ENV/bin/python" "$NODE_CODE/merge_grpo_adapter.py" \
  --base "$BASE" --adapter "$ADAPTER" --output "$NODE_MODEL" \
  --report "$ROOT/audit/merge-$SLURM_JOB_ID.json"
(cd "$NODE_MODEL" && sha256sum -c SHA256SUMS)
MODEL_SUMS_SHA=$(sha256sum "$NODE_MODEL/SHA256SUMS" | awk '{print $1}')

printf 'timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct\n' > "$ROOT/audit/gpu-$SLURM_JOB_ID.csv"
(
  while true; do
    stamp=$(date -Iseconds)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
      awk -v s="$stamp" -F', *' '{print s "," $1 "," $2 "," $3 "," $4}'
    sleep 10
  done
) >> "$ROOT/audit/gpu-$SLURM_JOB_ID.csv" &
gpu_sampler_pid=$!

HEADER_ROOT=/home/scc/bz22001004/runs/rollout-expand-20260824/vllm-python312-headers
export CPATH="$HEADER_ROOT/rootfs/usr/include:$HEADER_ROOT/rootfs/usr/include/python3.12:$HEADER_ROOT/rootfs/usr/include/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
export C_INCLUDE_PATH="$CPATH"

pids=()
for shard in 0 1 2 3; do
  shard_manifest="$ROOT/frozen4/shard_${shard}.jsonl"
  shard_sha=$(sha256sum "$shard_manifest" | awk '{print $1}')
  shard_out="$GENERATION/shard_${shard}"
  mkdir -p "$shard_out"
  srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:RTX5090:1 \
    "$VLLM_ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" generate-shard \
    --data "$DATA" --data-sha256 "$DATA_SHA" \
    --shard-manifest "$shard_manifest" --shard-sha256 "$shard_sha" \
    --model "$NODE_MODEL" --tokenizer-path "$TOKENIZER" \
    --model-sums "$NODE_MODEL/SHA256SUMS" --model-sums-sha256 "$MODEL_SUMS_SHA" \
    --output "$shard_out" --pass-k 32 \
    --generation-backend vllm --generation-batch-size 2 --generation-queue-depth 2 \
    --max-new-tokens 2048 --model-max-length 4096 --gpu-memory-utilization 0.9 \
    --expected-vllm-version 0.27.1 --expected-bitsandbytes-version 0.49.2 \
    --bitsandbytes-install-marker "$BNB_MARKER" \
    --expected-bitsandbytes-install-marker-sha256 1c92c3c58602661c741112eaa135965a5284cab96bf0855e22557baece0aee3f \
    --expected-bnb-loader-sha256 cf2c10fcaf5b6e997a8d5f80712af1f251a7292b5d76813e28b39bc03fa7c629 \
    --expected-bnb-linear-sha256 9817c93811bde5967ee9eed1efa551b8fc9e857e59fbd1b033d16c82ca42f112 \
    --temperature 1.0 --top-p 0.95 --seed 20260711 \
    > "$ROOT/logs/generate-shard${shard}-$SLURM_JOB_ID.out" \
    2> "$ROOT/logs/generate-shard${shard}-$SLURM_JOB_ID.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

kill "$gpu_sampler_pid" 2>/dev/null || true
wait "$gpu_sampler_pid" 2>/dev/null || true
gpu_sampler_pid=""

"$VLLM_ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" audit-generations \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GENERATION" --model-sums-sha256 "$MODEL_SUMS_SHA" \
  --output "$ROOT/audit/GENERATION_AUDIT.json"

ATTEMPTS=$(seq -s, 0 31)
if [[ ! -f "$COMPILE_MANIFEST/MANIFEST_FROZEN.json" ]]; then
  "$VLLM_ENV/bin/python" -P "$NODE_CODE/minif2f_split_verify.py" freeze-manifest \
    --grpo-entrypoint "$NODE_CODE/grpo_two_phase.py" \
    --grpo-entrypoint-sha256 a7247e82f37f0e63708e7df0f4341849140db73c29453d6d8a11253bd2d6f354 \
    --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
    --generation-root "$GENERATION" --model-sums-sha256 "$MODEL_SUMS_SHA" \
    --attempt-indices "$ATTEMPTS" \
    --receipt-origin compute_grpo_minif2f --execution-domain slurm_node_local \
    --output "$COMPILE_MANIFEST"
fi
COMPILE_MANIFEST_SHA=$(sha256sum "$COMPILE_MANIFEST/compile_manifest.jsonl" | awk '{print $1}')

source "$GRPO_RUN/stage_lean_olean_cache.sh"
"$VLLM_ENV/bin/python" -P "$NODE_CODE/minif2f_split_verify.py" verify \
  --grpo-entrypoint "$NODE_CODE/grpo_two_phase.py" \
  --grpo-entrypoint-sha256 a7247e82f37f0e63708e7df0f4341849140db73c29453d6d8a11253bd2d6f354 \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GENERATION" --model-sums-sha256 "$MODEL_SUMS_SHA" \
  --compile-manifest "$COMPILE_MANIFEST/compile_manifest.jsonl" \
  --compile-manifest-sha256 "$COMPILE_MANIFEST_SHA" \
  --receipt-origin compute_grpo_minif2f --execution-domain slurm_node_local \
  --output "$COMPILE_OUTPUT" --lean-project "$PROJECT/lean_project" \
  --task-spool-dir "$SPOOL" --workers 4 --timeout 180 --warmup-timeout 1200 \
  --queue-maxsize 256

python - <<'PY' "$ROOT" "$MODEL_SUMS_SHA" "$FROZEN_SHA"
import json,pathlib,sys
root=pathlib.Path(sys.argv[1])
g=json.loads((root/'audit/GENERATION_AUDIT.json').read_text())
c=json.loads((root/'compile/summary.json').read_text())
assert g['status']=='PASS' and g['problem_count']==244 and g['attempt_count']==7808
assert c['status']=='complete' and c['problem_count']==244 and c['attempt_count']==7808
payload={'schema':'minif2f_grpo_pass32_execution_complete_v1','status':'PASS',
 'problem_count':244,'attempt_count':7808,'split':'test','model_sums_sha256':sys.argv[2],
 'frozen_shards_sha256':sys.argv[3],'generation_audit':g,'compile_summary':c}
(root/'EXECUTION_COMPLETE.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY

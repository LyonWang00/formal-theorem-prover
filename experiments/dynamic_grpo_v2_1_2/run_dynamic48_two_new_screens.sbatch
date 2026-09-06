#!/usr/bin/env bash
#SBATCH --job-name=grpo-dyn48-2screen
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --open-mode=append

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1
PREVIOUS=/home/scc/bz22001004/runs/grpo-e2-adaptive-r6-comparable-lr1e5-ni2-20260904-v1
PROJECT=/home/scc/bz22001004/projects/formal-theorem-prover
BASE=$PROJECT/models/DeepSeek-Prover-V1.5-Base
SOURCE=/home/scc/bz22001004/runs/grpo-round6-old-sft-numiter2-lr1e5-20260904-v1/data_frozen/grpo_train2376_checkpoint9500_membership.jsonl
INITIAL_ADAPTER=/home/scc/bz22001004/runs/sft-resume-9344-to-18688-20260826/model/checkpoint-18688
TOKENIZER=/home/scc/bz22001004/runs/rollout-expand-20260824/contracts/deepseek-prover-v15-lean-whitespace-tokenizer-compat-v1
SFT_ENV=/home/scc/bz22001004/environments/formal-theorem-prover-sft
ENV=/home/scc/bz22001004/environments/rollout-vllm-cu130
STAGE=/home/scc/bz22001004/staging/grpo_adaptive_two_level_20260904
NODE_ROOT=/var/tmp/$USER-grpo-dyn48-$SLURM_JOB_ID
NODE_CODE=$NODE_ROOT/code
NODE_MODEL=$NODE_ROOT/merged-policy
SOCKET_DIR=$NODE_ROOT/pantograph-sockets
READY_DIR=$NODE_ROOT/pantograph-ready
SPOOL_DIR=$NODE_ROOT/pantograph-spool

mkdir -p "$ROOT/logs" "$ROOT/audit" "$NODE_CODE" "$SOCKET_DIR" "$READY_DIR" "$SPOOL_DIR"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=/home/scc/bz22001004/.elan/bin:$PATH
export OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PANTOGRAPH_POOL_SOCKET_DIR=$SOCKET_DIR
export PANTOGRAPH_POOL_WORKERS=4
export PANTOGRAPH_POOL_AUTHKEY="dyn48-${SLURM_JOB_ID}-allocation-local"

service_pids=()
cleanup() {
  set +e
  for pid in "${service_pids[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  for pid in "${service_pids[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  case "$(readlink -f "$NODE_ROOT")" in
    (/var/tmp/"$USER"-grpo-dyn48-"$SLURM_JOB_ID") rm -rf -- "$NODE_ROOT" ;;
    (*) echo "unsafe node root; refusing cleanup" >&2 ;;
  esac
}
trap cleanup EXIT INT TERM

tar -C "$PROJECT" --exclude='lean_prover/Dataset' --exclude='__pycache__' --exclude='*.pyc' -cf - lean_prover | tar -C "$NODE_CODE" -xf -
tar -C "$NODE_CODE" -xf /home/scc/bz22001004/runs/minif2f-resume18688-pass32-20260827/staging/lean_training_runtime.tar
cp "$ROOT/code/grpo_two_phase_bf16.py" "$NODE_CODE/grpo_two_phase.py"
cp "$ROOT/code/pool_hard_timeout_e2.py" "$NODE_CODE/lean_prover/lean_training/verification/pool.py"
cp "$ROOT/code/persistent_pantograph_service.py" "$NODE_CODE/persistent_pantograph_service.py"
cp "$ROOT/code/merge_grpo_adapter.py" "$NODE_CODE/merge_adapter.py"
export PYTHONPATH=$NODE_CODE

source /home/scc/bz22001004/runs/grpo-frozen8758-initialparams-20260901-v1/stage_lean_olean_cache.sh
exec 9>&-

for worker in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES= "$ENV/bin/python" -P "$NODE_CODE/persistent_pantograph_service.py" \
    --socket "$SOCKET_DIR/worker-$worker.sock" \
    --ready-file "$READY_DIR/worker-$worker.json" --worker-index "$worker" \
    --lean-project "$PROJECT/lean_project" --task-spool-dir "$SPOOL_DIR/worker-$worker" \
    --timeout 60 --warmup-timeout 1800 --queue-maxsize 256 \
    > "$ROOT/logs/persistent-worker${worker}-$SLURM_JOB_ID.out" \
    2> "$ROOT/logs/persistent-worker${worker}-$SLURM_JOB_ID.err" &
  service_pids+=("$!")
  deadline=$((SECONDS + 2100))
  while [[ ! -s "$READY_DIR/worker-$worker.json" ]]; do
    kill -0 "${service_pids[-1]}" 2>/dev/null || { echo "worker $worker exited during warmup" >&2; exit 1; }
    (( SECONDS < deadline )) || { echo "worker $worker warmup deadline exceeded" >&2; exit 1; }
    sleep 2
  done
done

python3 - "$READY_DIR" "$ROOT/audit/persistent-workers-$SLURM_JOB_ID.json" <<'PY'
import json,pathlib,sys,time
ready=pathlib.Path(sys.argv[1]); output=pathlib.Path(sys.argv[2])
rows=[json.loads((ready/f'worker-{i}.json').read_text()) for i in range(4)]
assert all(row['status']=='ready' for row in rows)
output.write_text(json.dumps({'schema':'dynamic48_persistent_workers_v1','status':'READY','workers':rows,'recorded_at_unix':time.time()},indent=2,sort_keys=True)+'\n')
PY

sha_file() { sha256sum "$1" | awk '{print $1}'; }

merge_policy() {
  local adapter=$1 cycle=$2
  rm -rf "$NODE_MODEL"
  "$SFT_ENV/bin/python" "$NODE_CODE/merge_adapter.py" \
    --base "$BASE" --adapter "$adapter" --output "$NODE_MODEL" --report "$cycle/merge.json" >&2
  (cd "$NODE_MODEL" && sha256sum -c SHA256SUMS) >&2
  sha_file "$NODE_MODEL/SHA256SUMS"
}

assert_unseen_screen() {
  local pool=$1 history=$2
  "$ENV/bin/python" - "$pool" "$history" <<'PY'
import json,pathlib,sys
def rows(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]
def rid(row): return str(row.get('id', row.get('problem_id')))
pool={rid(row) for row in rows(sys.argv[1])}
seen={str(row['problem_id']) for row in rows(sys.argv[2])}
overlap=pool & seen
assert not overlap, f'screen unexpectedly reused {len(overlap)} previously visited problems'
PY
}

generate_stage() {
  local cycle=$1 stage=$2 data=$3 pass_k=$4 seed=$5 model_sha=$6
  local out=$cycle/$stage n data_sha frozen frozen_sha cycle_name staging
  cycle_name=${cycle%/frozen}
  cycle_name=${cycle_name##*/}
  n=$(wc -l < "$data")
  test "$n" -gt 0
  mkdir -p "$out" "$out/generation"
  staging=$out/generation_staging.jsonl
  "$ENV/bin/python" - "$data" "$staging" <<'PY'
import json, pathlib, sys
source, target = map(pathlib.Path, sys.argv[1:])
rows=[]
for line in source.read_text(encoding='utf-8').splitlines():
    if not line.strip():
        continue
    row=json.loads(line)
    prompt=str(row['prompt'])
    if prompt.endswith(':= by sorry\n'):
        pass
    elif prompt.endswith(':=\n'):
        row['prompt']=prompt[:-1]+' by sorry\n'
    else:
        raise ValueError(f"{row.get('id', row.get('problem_id'))}: unsupported prompt suffix {prompt[-32:]!r}")
    rows.append(row)
with target.open('w',encoding='utf-8',newline='\n') as handle:
    for row in rows:
        handle.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
assert len(rows)>0
PY
  test "$(wc -l < "$staging")" -eq "$n"
  data_sha=$(sha_file "$staging")
  frozen=$out/frozen4/SHARDS_FROZEN.json
  if [[ -f "$frozen" ]]; then
    "$ENV/bin/python" - "$frozen" "$data_sha" "$n" "$pass_k" "$seed" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); data_sha=sys.argv[2]; n,k,seed=map(int,sys.argv[3:6])
s=json.loads(p.read_text())
assert s['status']=='frozen' and s['data_sha256']==data_sha
assert s['problem_count']==n and s['expected_attempt_count']==n*k
assert s['attempts_per_problem']==k and s['sampling_seed_base']==seed
assert s['shard_count']==4
for item in s['shards']:
    shard=p.parent/item['path']
    assert shard.exists()
    assert hashlib.sha256(shard.read_bytes()).hexdigest()==item['sha256']
print(f"REUSE_FROZEN_GENERATION_SHARDS path={p}")
PY
  else
    "$ENV/bin/python" "$STAGE/freeze_stage_shards.py" \
      --data "$staging" --output "$out/frozen4" --tokenizer "$TOKENIZER" \
      --shards 4 --pass-k "$pass_k" --max-new-tokens 512 --max-model-len 1024 \
      --generation-batch-size 4 --seed "$seed"
  fi
  frozen_sha=$(sha_file "$frozen")
  mkdir -p "$ROOT/cache/flashinfer-autotune"
  if [[ -f "$out/verification/summary.json" ]]; then
    if "$ENV/bin/python" - "$out/verification/summary.json" "$frozen_sha" "$model_sha" "$n" "$pass_k" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); frozen_sha,model_sha=sys.argv[2:4]; n,k=map(int,sys.argv[4:6])
s=json.loads(p.read_text())
assert s['status']=='complete'
assert s['frozen_shards_sha256']==frozen_sha
assert s['model_sums_sha256']==model_sha
assert s['problem_count']==n and s['attempt_count']==n*k
PY
    then
      echo "REUSE_COMPLETE_STAGE cycle=$cycle_name stage=$stage attempts=$((n * pass_k))"
      return
    fi
    echo "invalid existing verification summary: $out/verification/summary.json" >&2
    return 1
  fi
  pids=()
  local serial_generation_warmup_done=false
  for shard in 0 1 2 3; do
    local manifest manifest_sha
    manifest=$out/frozen4/shard_${shard}.jsonl
    manifest_sha=$(sha_file "$manifest")
    mkdir -p "$out/generation/shard_${shard}"
    if [[ -f "$out/generation/shard_${shard}/GENERATION_COMPLETE.json" ]]; then
      if "$ENV/bin/python" - "$out/generation/shard_${shard}/GENERATION_COMPLETE.json" \
          "$manifest_sha" "$data_sha" "$model_sha" "$pass_k" <<'PY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]); manifest_sha,data_sha,model_sha=sys.argv[2:5]; k=int(sys.argv[5])
s=json.loads(p.read_text())
assert s['status']=='complete'
assert s['shard_manifest_sha256']==manifest_sha
assert s['data_sha256']==data_sha
assert s['model_sums_sha256']==model_sha
assert int(s['pass_k'])==k
assert int(s['attempt_count'])==int(s['problem_count'])*k
PY
      then
        echo "REUSE_COMPLETE_GENERATION_SHARD cycle=$cycle_name stage=$stage shard=$shard"
        continue
      fi
      echo "invalid generation completion marker: $out/generation/shard_${shard}/GENERATION_COMPLETE.json" >&2
      return 1
    fi
    if [[ "$serial_generation_warmup_done" == false ]]; then
      echo "SERIAL_GENERATION_WARMUP cycle=$cycle_name stage=$stage shard=$shard"
      srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:RTX5090:1 \
        env VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$ROOT/cache/flashinfer-autotune/shard-$shard" \
        "$ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" generate-shard \
        --data "$staging" --data-sha256 "$data_sha" \
        --shard-manifest "$manifest" --shard-sha256 "$manifest_sha" \
        --model "$NODE_MODEL" --tokenizer-path "$TOKENIZER" \
        --model-sums "$NODE_MODEL/SHA256SUMS" --model-sums-sha256 "$model_sha" \
        --output "$out/generation/shard_${shard}" --pass-k "$pass_k" \
        --generation-backend vllm --generation-batch-size 4 --generation-queue-depth 8 \
        --max-new-tokens 512 --model-max-length 1024 --gpu-memory-utilization 0.88 \
        --expected-vllm-version 0.27.1 --temperature 1.0 --top-p 0.95 --seed "$seed" \
        > "$ROOT/logs/${cycle_name}-${stage}-generate-shard${shard}-$SLURM_JOB_ID.out" \
        2> "$ROOT/logs/${cycle_name}-${stage}-generate-shard${shard}-$SLURM_JOB_ID.err"
      serial_generation_warmup_done=true
    else
      srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:RTX5090:1 \
        env VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$ROOT/cache/flashinfer-autotune/shard-$shard" \
        "$ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" generate-shard \
        --data "$staging" --data-sha256 "$data_sha" \
        --shard-manifest "$manifest" --shard-sha256 "$manifest_sha" \
        --model "$NODE_MODEL" --tokenizer-path "$TOKENIZER" \
        --model-sums "$NODE_MODEL/SHA256SUMS" --model-sums-sha256 "$model_sha" \
        --output "$out/generation/shard_${shard}" --pass-k "$pass_k" \
        --generation-backend vllm --generation-batch-size 4 --generation-queue-depth 8 \
        --max-new-tokens 512 --model-max-length 1024 --gpu-memory-utilization 0.88 \
        --expected-vllm-version 0.27.1 --temperature 1.0 --top-p 0.95 --seed "$seed" \
        > "$ROOT/logs/${cycle_name}-${stage}-generate-shard${shard}-$SLURM_JOB_ID.out" \
        2> "$ROOT/logs/${cycle_name}-${stage}-generate-shard${shard}-$SLURM_JOB_ID.err" &
      pids+=("$!")
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  test "$(find "$out/generation" -type f -name 'problem_*.jsonl' -exec cat {} + | wc -l)" -eq "$((n * pass_k))"
  export PANTOGRAPH_POOL_CLIENT_MODE=global
  "$ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" verify \
    --frozen-shards "$frozen" --frozen-shards-sha256 "$frozen_sha" \
    --generation-root "$out/generation" --model-sums-sha256 "$model_sha" \
    --output "$out/verification" --lean-project "$PROJECT/lean_project" \
    --task-spool-dir "$SPOOL_DIR/${cycle##*/}-$stage" \
    --timeout 60 --warmup-timeout 1800 --queue-maxsize 256 --verify-batch-size 256
  "$ENV/bin/python" - "$out/verification/summary.json" "$n" "$pass_k" <<'PY'
import json,sys
s=json.load(open(sys.argv[1])); n=int(sys.argv[2]); k=int(sys.argv[3])
assert s['status']=='complete' and s['problem_count']==n and s['attempt_count']==n*k, s
PY
}

bind_stage() {
  local cycle=$1 stage=$2 pool=$3 generation=$4 receipts=$5 output=$6
  if [[ -f "$output" && -f "${output%.jsonl}.audit.json" ]]; then
    "$ENV/bin/python" - "$cycle/CYCLE_FROZEN.json" "$stage" "$pool" "$receipts" \
      "$output" "${output%.jsonl}.audit.json" <<'PY'
import hashlib,json,pathlib,sys
manifest,stage,pool,raw,output,audit=sys.argv[1:]
manifest=json.loads(pathlib.Path(manifest).read_text())
report=json.loads(pathlib.Path(audit).read_text())
sha=lambda p: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
lines=lambda p: sum(1 for line in pathlib.Path(p).open() if line.strip())
assert report['schema']=='dynamic_grpo_bound_receipts_v2'
assert report['stage']==stage and report['cycle_id']==manifest['cycle_id']
assert report['old_policy_sha256']==manifest['old_policy_sha256']
assert report['selection_manifest_sha256']==manifest['screen_pool_sha256']
assert report['raw_receipts_sha256']==sha(raw)
assert report['bound_receipts_sha256']==sha(output)
assert report['problem_count']==lines(pool)
assert report['receipt_count']==lines(output)
print(f"REUSE_BOUND_RECEIPTS stage={stage} path={output}")
PY
    return
  fi
  if [[ -e "$output" || -e "${output%.jsonl}.audit.json" ]]; then
    echo "incomplete bound-receipt artifact pair for $output" >&2
    return 1
  fi
  "$ENV/bin/python" "$ROOT/code/dynamic_cycle_controller.py" bind-receipts \
    --cycle-dir "$cycle" --stage "$stage" --stage-pool "$pool" \
    --generation-root "$generation" --raw-receipts "$receipts" --output "$output"
}

plan_topup_stage() {
  local cycle=$1 stage=$2 output=$3 seed=$4
  shift 4
  local marker=$output/TOPUP_FROZEN.json
  if [[ -f "$marker" ]]; then
    "$ENV/bin/python" - "$marker" "$cycle" "$stage" "$output" "$@" <<'PY'
import hashlib,json,pathlib,sys
marker,cycle,stage,output=map(pathlib.Path,sys.argv[1:5]); receipts=[pathlib.Path(x) for x in sys.argv[5:]]
s=json.loads(marker.read_text())
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
pool=output/f'{stage.name}_pool.jsonl'
assert s['schema']=='dynamic_grpo_topup_plan_v2'
assert s['stage']==stage.name
assert s['pool_sha256']==sha(pool)
assert s['bound_receipt_sha256']==[sha(p) for p in receipts]
print(f'REUSE_TOPUP_PLAN stage={stage.name} path={marker}')
PY
    return
  fi
  "$ENV/bin/python" "$ROOT/code/dynamic_cycle_controller.py" plan-topup \
    --cycle-dir "$cycle" --stage "$stage" --bound-receipts "$@" \
    --output-dir "$output" --seed "$seed"
}

prepare_fresh_cycle() {
  local index=$1 adapter=$2 history=$3 stop=$4 seed=$5
  local cycle=$ROOT/cycle-$index policy_sha rebuilt_policy_sha
  if [[ -f "$ROOT/model/checkpoint-$stop/trainer_state.json" ]]; then
    "$ENV/bin/python" - "$ROOT/model/checkpoint-$stop/trainer_state.json" "$stop" <<'PY'
import json,pathlib,sys
state=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert int(state['global_step'])==int(sys.argv[2]), state['global_step']
PY
    test -f "$cycle/frozen/schedule/CYCLE_SCHEDULE_FROZEN.json"
    return
  fi
  if [[ -f "$cycle/frozen/schedule/CYCLE_SCHEDULE_FROZEN.json" ]]; then
    policy_sha=$("$ENV/bin/python" - "$cycle" "$adapter" "$history" "$index" <<'PY'
import hashlib,json,pathlib,sys
cycle,adapter,history=map(pathlib.Path,sys.argv[1:4]); index=int(sys.argv[4])
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
merge=json.loads((cycle/'merge.json').read_text())
frozen=json.loads((cycle/'frozen/CYCLE_FROZEN.json').read_text())
schedule=json.loads((cycle/'frozen/schedule/CYCLE_SCHEDULE_FROZEN.json').read_text())
assert pathlib.Path(merge['adapter']).resolve()==adapter.resolve()
assert merge['adapter_model_sha256']==sha(adapter/'adapter_model.safetensors')
assert frozen['status']=='FROZEN'
assert frozen['cycle_id']==f'dynamic48-cycle-{index}'
assert frozen['history_sha256']==sha(history)
assert schedule['cycle_identity']['cycle_id']==f'dynamic48-cycle-{index}'
assert schedule['cycle_identity']['old_policy_sha256']==merge['model_sums_sha256']
assert schedule['train_schedule_sha256']==sha(cycle/'frozen/schedule/train_schedule.jsonl')
for stage,expected in (('pass8',4096),('pass16',1024),('pass32',1024)):
    summary=json.loads((cycle/f'frozen/{stage}/verification/summary.json').read_text())
    assert summary['status']=='complete' and summary['attempt_count']==expected, (stage,summary)
print(merge['model_sums_sha256'])
PY
)
    ROOT="$ROOT" CYCLE_DIR="$cycle/frozen" ADAPTER="$adapter" CYCLE_ID="dynamic48-cycle-$index" \
      OLD_POLICY_SHA256="$policy_sha" CYCLE_STOP_GLOBAL_STEP="$stop" FULL_MAX_STEPS="${FULL_RUN_MAX_STEPS:-48}" \
      RESUME_CHECKPOINT="$adapter" bash "$ROOT/code/train_dynamic_cycle.template.sh"
    return
  fi
  mkdir -p "$cycle"
  if [[ -f "$cycle/frozen/CYCLE_FROZEN.json" ]]; then
    policy_sha=$("$ENV/bin/python" - "$cycle" "$adapter" "$history" "$index" <<'PY'
import hashlib,json,pathlib,sys
cycle,adapter,history=map(pathlib.Path,sys.argv[1:4]); index=int(sys.argv[4])
sha=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
merge=json.loads((cycle/'merge.json').read_text())
frozen=json.loads((cycle/'frozen/CYCLE_FROZEN.json').read_text())
assert pathlib.Path(merge['adapter']).resolve()==adapter.resolve()
assert merge['adapter_model_sha256']==sha(adapter/'adapter_model.safetensors')
assert frozen['status']=='FROZEN' and frozen['cycle_id']==f'dynamic48-cycle-{index}'
assert frozen['history_sha256']==sha(history)
assert frozen['old_policy_sha256']==merge['model_sums_sha256']
print(merge['model_sums_sha256'])
PY
)
    rebuilt_policy_sha=$(merge_policy "$adapter" "$cycle")
    test "$rebuilt_policy_sha" = "$policy_sha"
    echo "REUSE_FROZEN_CYCLE cycle=dynamic48-cycle-$index"
  else
    policy_sha=$(merge_policy "$adapter" "$cycle")
    "$ENV/bin/python" "$ROOT/code/dynamic_cycle_controller.py" freeze \
      --source-dataset "$SOURCE" --history "$history" --cycle-id "dynamic48-cycle-$index" \
      --old-policy-sha256 "$policy_sha" --output-dir "$cycle/frozen" --seed "$seed"
  fi
  cycle=$cycle/frozen
  assert_unseen_screen "$cycle/screen_pool.jsonl" "$history"
  generate_stage "$cycle" pass8 "$cycle/screen_pool.jsonl" 8 "$seed" "$policy_sha"
  bind_stage "$cycle" pass8 "$cycle/screen_pool.jsonl" "$cycle/pass8/generation" \
    "$cycle/pass8/verification/receipts.jsonl" "$cycle/pass8.bound.jsonl"
  plan_topup_stage "$cycle" pass16 "$cycle/pass16-plan" "$((seed + 1))" \
    "$cycle/pass8.bound.jsonl"
  generate_stage "$cycle" pass16 "$cycle/pass16-plan/pass16_pool.jsonl" 8 "$((seed + 1))" "$policy_sha"
  bind_stage "$cycle" pass16 "$cycle/pass16-plan/pass16_pool.jsonl" "$cycle/pass16/generation" \
    "$cycle/pass16/verification/receipts.jsonl" "$cycle/pass16.bound.jsonl"
  plan_topup_stage "$cycle" pass32 "$cycle/pass32-plan" "$((seed + 2))" \
    "$cycle/pass8.bound.jsonl" "$cycle/pass16.bound.jsonl"
  generate_stage "$cycle" pass32 "$cycle/pass32-plan/pass32_pool.jsonl" 16 "$((seed + 2))" "$policy_sha"
  bind_stage "$cycle" pass32 "$cycle/pass32-plan/pass32_pool.jsonl" "$cycle/pass32/generation" \
    "$cycle/pass32/verification/receipts.jsonl" "$cycle/pass32.bound.jsonl"
  "$ENV/bin/python" "$ROOT/code/dynamic_cycle_controller.py" finalize \
    --cycle-dir "$cycle" \
    --bound-receipts "$cycle/pass8.bound.jsonl" "$cycle/pass16.bound.jsonl" "$cycle/pass32.bound.jsonl" \
    --output-dir "$cycle/schedule" --history "$history" --seed "$((seed + 3))"
  ROOT="$ROOT" CYCLE_DIR="$cycle" ADAPTER="$adapter" CYCLE_ID="dynamic48-cycle-$index" \
    OLD_POLICY_SHA256="$policy_sha" CYCLE_STOP_GLOBAL_STEP="$stop" FULL_MAX_STEPS="${FULL_RUN_MAX_STEPS:-48}" \
    RESUME_CHECKPOINT="$adapter" bash "$ROOT/code/train_dynamic_cycle.template.sh"
}

# Cycle 0 reuses the exact previous 512-problem old-SFT screen and capped top-ups.
# A failed later cycle may resume from the complete checkpoint-16 without
# regenerating or retraining this immutable cycle.
C0=$ROOT/cycle-0
if [[ ! -f "$ROOT/model/checkpoint-16/trainer_state.json" ]]; then
  OLD_SHA=$(python3 -c 'import json; print(json.load(open("'$PREVIOUS'/screen_pass8_v3/merge.json"))["model_sums_sha256"])')
  test "$OLD_SHA" = f97bcc5cd13bac535202774c0f2e02909665621e9dc3329edfd23d74909214ae
  test "$(sha_file "$INITIAL_ADAPTER/adapter_model.safetensors")" = d7db54435255265eaa305d65c3045a4fa1420e802798757ad59550ef6dc6c296
  "$ENV/bin/python" "$ROOT/code/dynamic_cycle_controller.py" freeze \
    --source-dataset "$PREVIOUS/data_frozen/rollout_pool_512.jsonl" \
    --cycle-id dynamic48-cycle-0 --old-policy-sha256 "$OLD_SHA" --output-dir "$C0" --seed 20260910
  bind_stage "$C0" pass8 "$C0/screen_pool.jsonl" "$PREVIOUS/screen_pass8_v3/generation" \
    "$PREVIOUS/screen_pass8_v3/verification/receipts.jsonl" "$C0/pass8.bound.jsonl"
  bind_stage "$C0" pass16 "$PREVIOUS/topup_selection_pass16_v1/pass16_topup_pool_128.jsonl" \
    "$PREVIOUS/topup_pass16_v1/generation" "$PREVIOUS/topup_pass16_v1/verification/receipts.jsonl" "$C0/pass16.bound.jsonl"
  bind_stage "$C0" pass32 "$PREVIOUS/topup_selection_pass32_v1/pass32_topup_pool_64.jsonl" \
    "$PREVIOUS/topup_pass32_v1/generation" "$PREVIOUS/topup_pass32_v1/verification/receipts.jsonl" "$C0/pass32.bound.jsonl"
  "$ENV/bin/python" "$ROOT/code/dynamic_cycle_controller.py" finalize \
    --cycle-dir "$C0" --bound-receipts "$C0/pass8.bound.jsonl" "$C0/pass16.bound.jsonl" "$C0/pass32.bound.jsonl" \
    --output-dir "$C0/schedule" --seed 20260913
  ROOT="$ROOT" CYCLE_DIR="$C0" ADAPTER="$INITIAL_ADAPTER" CYCLE_ID=dynamic48-cycle-0 \
    OLD_POLICY_SHA256="$OLD_SHA" CYCLE_STOP_GLOBAL_STEP=16 FULL_MAX_STEPS=48 \
    bash "$ROOT/code/train_dynamic_cycle.template.sh"
else
  "$ENV/bin/python" - "$ROOT/model/checkpoint-16/trainer_state.json" <<'PY'
import json,sys
state=json.load(open(sys.argv[1])); assert int(state['global_step'])==16, state['global_step']
PY
  test -f "$C0/schedule/rollout_history.jsonl"
  test "$(sha_file "$ROOT/model/checkpoint-16/adapter_model.safetensors")" = f22d59096f87782b4c0cb36051858010c4c329090724b70b796a73665e84a84e
  test "$(sha_file "$INITIAL_ADAPTER/adapter_model.safetensors")" = d7db54435255265eaa305d65c3045a4fa1420e802798757ad59550ef6dc6c296
fi

prepare_fresh_cycle 1 "$ROOT/model/checkpoint-16" "$C0/schedule/rollout_history.jsonl" 32 20260920
prepare_fresh_cycle 2 "$ROOT/model/checkpoint-32" "$ROOT/cycle-1/frozen/schedule/rollout_history.jsonl" 48 20260930

test -f "$ROOT/model/checkpoint-48/trainer_state.json"
python3 - "$ROOT" "$SLURM_JOB_ID" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); checkpoint=root/'model/checkpoint-48'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert int(state['global_step'])==48
payload={
  'schema':'dynamic_adaptive_grpo_48step_complete_v1','status':'PASS',
  'global_step':48,'slurm_job_id':int(sys.argv[2]),'checkpoint':str(checkpoint),
  'adapter_model_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'new_512_problem_screens':2,'completed_at_unix':time.time()
}
(root/'TRAINING_COMPLETE.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY

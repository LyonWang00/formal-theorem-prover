#!/usr/bin/env bash
#SBATCH --job-name=m2f-sft75344-lr2e4-p32
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/scc/bz22001004/runs/sft-main75344-deepseekprover-r32-lr2e4-epoch2-20260903-v1/logs/eval-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/sft-main75344-deepseekprover-r32-lr2e4-epoch2-20260903-v1/logs/eval-%j.err

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/sft-main75344-deepseekprover-r32-lr2e4-epoch2-20260903-v1
EVAL=$ROOT/eval_pass32
PROJECT=/home/scc/bz22001004/projects/formal-theorem-prover
BASE=$PROJECT/models/DeepSeek-Prover-V1.5-Base
ADAPTER=$ROOT/model
SFT_ENV=/home/scc/bz22001004/environments/formal-theorem-prover-sft
VLLM_ENV=/home/scc/bz22001004/environments/rollout-vllm-cu130
BASELINE=/home/scc/bz22001004/runs/minif2f-resume18688-pass32-20260827
REFERENCE=/home/scc/bz22001004/runs/minif2f-grpo-frozen8758-pass32-20260901-v1
GRPO_RUN=/home/scc/bz22001004/runs/grpo-frozen8758-initialparams-20260901-v1
TOKENIZER=/home/scc/bz22001004/runs/rollout-expand-20260824/contracts/deepseek-prover-v15-lean-whitespace-tokenizer-compat-v1
BNB_ROOT=/home/scc/bz22001004/runs/rollout-expand-20260824/vllm-bitsandbytes-0.49.2
DATA=$BASELINE/input/full_test244_data.jsonl
MANIFEST=$BASELINE/input/full_test244_manifest.jsonl
FROZEN=$REFERENCE/frozen4/SHARDS_FROZEN.json
GENERATION=$EVAL/generation
COMPILE_MANIFEST=$EVAL/compile_manifest_all32
COMPILE_OUTPUT=$EVAL/compile_persistent_v1
NODE_ROOT=/var/tmp/$USER-sft75344-lr2e4-pass32-$SLURM_JOB_ID
NODE_CODE=$NODE_ROOT/code
NODE_MODEL=$NODE_ROOT/merged-sft75344
SPOOL=$NODE_ROOT/spool

test -f "$ROOT/TRAINING_COMPLETE.json"
test ! -e "$EVAL/EXECUTION_COMPLETE.json"
mkdir -p "$ROOT/logs" "$EVAL/audit" "$GENERATION" "$COMPILE_OUTPUT" "$NODE_CODE" "$SPOOL"

gpu_sampler_pid=""
cleanup() {
  set +e
  if [[ -n "$gpu_sampler_pid" ]]; then kill "$gpu_sampler_pid" 2>/dev/null || true; fi
  case "$(readlink -f "$NODE_ROOT")" in
    (/var/tmp/"$USER"-sft75344-lr2e4-pass32-"$SLURM_JOB_ID") rm -rf -- "$NODE_ROOT" ;;
    (*) echo "unsafe node root; refusing cleanup" >&2 ;;
  esac
}
trap cleanup EXIT INT TERM

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PATH=/home/scc/bz22001004/.elan/bin:$PATH
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=4

tar -C "$PROJECT" --exclude='lean_prover/Dataset' --exclude='__pycache__' --exclude='*.pyc' -cf - lean_prover | tar -C "$NODE_CODE" -xf -
tar -C "$NODE_CODE" -xf "$BASELINE/staging/lean_training_runtime.tar"
cp "$BASELINE/staging/grpo_two_phase.py" "$NODE_CODE/grpo_two_phase.py"
cp "$BASELINE/staging/grpo_vllm_env_wrapper.py" "$NODE_CODE/grpo_vllm_env_wrapper.py"
cp "$REFERENCE/scripts/minif2f_full_split_verify_nowarm.py" "$NODE_CODE/minif2f_split_verify.py"
cp "$REFERENCE/scripts/merge_grpo_adapter.py" "$NODE_CODE/merge_adapter.py"
cp "$REFERENCE/scripts/analyze_corrected.py" "$NODE_CODE/analyze_corrected.py"
export PYTHONPATH=$NODE_CODE

test "$(sha256sum "$NODE_CODE/minif2f_split_verify.py" | awk '{print $1}')" = 811e67b39b3726a4808b9e189e39fde0a80ed14da983cbb3990b198645a94d49
test "$(sha256sum "$NODE_CODE/merge_adapter.py" | awk '{print $1}')" = 98449ef892559d63ab2595528ff0f48d54e64260faad004b025c6cd4e4962aff
test "$(sha256sum "$NODE_CODE/analyze_corrected.py" | awk '{print $1}')" = bf3fee20ebf8119b750ef57f2085cf2c824adb03c1b508a23fe8e36539b7a98d

python3 - "$DATA" "$ADAPTER" <<'PY'
import json,pathlib,sys
data,adapter=map(pathlib.Path,sys.argv[1:])
rows=[json.loads(x) for x in data.open() if x.strip()]
assert len(rows)==244 and {r.get('split') for r in rows}=={'test'}
assert len({r['id'] for r in rows})==244
assert not any(any(k in r for k in ('proof','completion','reference_proof')) for r in rows)
state=json.loads((adapter/'trainer_state.json').read_text())
config=json.loads((adapter/'training_config.json').read_text())['config']
ac=json.loads((adapter/'adapter_config.json').read_text())
assert state['global_step']==18836 and abs(float(state['epoch'])-2.0)<1e-9
assert config['learning_rate']==2e-4 and config['lora_r']==32 and config['lora_alpha']==64
assert ac['r']==32 and ac['lora_alpha']==64
print('EVAL_GATE PASS test244 adapter_step18836 lr2e-4 r32 alpha64')
PY

DATA_SHA=$(sha256sum "$DATA" | awk '{print $1}')
FROZEN_SHA=$(sha256sum "$FROZEN" | awk '{print $1}')
test "$DATA_SHA" = 2a261e530c74c619507d4efd1b4e42cf7f1c7a19c8458880f223d61b41a02d24

"$SFT_ENV/bin/python" "$NODE_CODE/merge_adapter.py" --base "$BASE" --adapter "$ADAPTER" \
  --output "$NODE_MODEL" --report "$EVAL/audit/merge-$SLURM_JOB_ID.json"
(cd "$NODE_MODEL" && sha256sum -c SHA256SUMS)
MODEL_SUMS_SHA=$(sha256sum "$NODE_MODEL/SHA256SUMS" | awk '{print $1}')

printf 'timestamp,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_pct\n' > "$EVAL/audit/gpu-$SLURM_JOB_ID.csv"
(
  while true; do
    stamp=$(date -Iseconds)
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits |
      awk -v s="$stamp" -F', *' '{print s "," $1 "," $2 "," $3 "," $4}'
    sleep 10
  done
) >> "$EVAL/audit/gpu-$SLURM_JOB_ID.csv" &
gpu_sampler_pid=$!

HEADER_ROOT=/home/scc/bz22001004/runs/rollout-expand-20260824/vllm-python312-headers
export CPATH="$HEADER_ROOT/rootfs/usr/include:$HEADER_ROOT/rootfs/usr/include/python3.12:$HEADER_ROOT/rootfs/usr/include/x86_64-linux-gnu/python3.12${CPATH:+:$CPATH}"
export C_INCLUDE_PATH=$CPATH

pids=()
for shard in 0 1 2 3; do
  shard_manifest="$REFERENCE/frozen4/shard_${shard}.jsonl"
  shard_sha=$(sha256sum "$shard_manifest" | awk '{print $1}')
  shard_out="$GENERATION/shard_${shard}"
  mkdir -p "$shard_out"
  srun --exclusive --ntasks=1 --cpus-per-task=4 --gres=gpu:RTX5090:1 \
    "$VLLM_ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" generate-shard \
    --data "$DATA" --data-sha256 "$DATA_SHA" --shard-manifest "$shard_manifest" --shard-sha256 "$shard_sha" \
    --model "$NODE_MODEL" --tokenizer-path "$TOKENIZER" \
    --model-sums "$NODE_MODEL/SHA256SUMS" --model-sums-sha256 "$MODEL_SUMS_SHA" \
    --output "$shard_out" --pass-k 32 --generation-backend vllm --generation-batch-size 2 --generation-queue-depth 2 \
    --max-new-tokens 2048 --model-max-length 4096 --gpu-memory-utilization 0.9 \
    --expected-vllm-version 0.27.1 --expected-bitsandbytes-version 0.49.2 \
    --bitsandbytes-install-marker "$BNB_ROOT/INSTALL_COMPLETE.json" \
    --expected-bitsandbytes-install-marker-sha256 1c92c3c58602661c741112eaa135965a5284cab96bf0855e22557baece0aee3f \
    --expected-bnb-loader-sha256 cf2c10fcaf5b6e997a8d5f80712af1f251a7292b5d76813e28b39bc03fa7c629 \
    --expected-bnb-linear-sha256 9817c93811bde5967ee9eed1efa551b8fc9e857e59fbd1b033d16c82ca42f112 \
    --temperature 1.0 --top-p 0.95 --seed 20260711 \
    > "$ROOT/logs/generate-shard${shard}-$SLURM_JOB_ID.out" 2> "$ROOT/logs/generate-shard${shard}-$SLURM_JOB_ID.err" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

kill "$gpu_sampler_pid" 2>/dev/null || true
wait "$gpu_sampler_pid" 2>/dev/null || true
gpu_sampler_pid=""

"$VLLM_ENV/bin/python" -P "$NODE_CODE/grpo_two_phase.py" audit-generations \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GENERATION" --model-sums-sha256 "$MODEL_SUMS_SHA" --output "$EVAL/audit/GENERATION_AUDIT.json"

ATTEMPTS=$(seq -s, 0 31)
"$VLLM_ENV/bin/python" -P "$NODE_CODE/minif2f_split_verify.py" freeze-manifest \
  --grpo-entrypoint "$NODE_CODE/grpo_two_phase.py" \
  --grpo-entrypoint-sha256 a7247e82f37f0e63708e7df0f4341849140db73c29453d6d8a11253bd2d6f354 \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GENERATION" --model-sums-sha256 "$MODEL_SUMS_SHA" --attempt-indices "$ATTEMPTS" \
  --receipt-origin compute_sft75344_lr2e4_minif2f --execution-domain slurm_node_local --output "$COMPILE_MANIFEST"
COMPILE_MANIFEST_SHA=$(sha256sum "$COMPILE_MANIFEST/compile_manifest.jsonl" | awk '{print $1}')

source "$GRPO_RUN/stage_lean_olean_cache.sh"
OLEAN_CACHE_ROOT=/tmp/$USER/formal-theorem-prover-olean-v4.29.1
CACHE_PREWARM_START=$(date +%s)
CACHE_PREWARM_BYTES=$(find "$OLEAN_CACHE_ROOT" -type f -printf '%s\n' | awk '{total += $1} END {printf "%.0f", total}')
echo "OLEAN_PAGE_PREWARM_START root=$OLEAN_CACHE_ROOT bytes=$CACHE_PREWARM_BYTES"
find "$OLEAN_CACHE_ROOT" -type f -exec cat {} + >/dev/null
echo "OLEAN_PAGE_PREWARM_COMPLETE seconds=$(($(date +%s)-CACHE_PREWARM_START))"

"$VLLM_ENV/bin/python" -P "$NODE_CODE/minif2f_split_verify.py" verify \
  --grpo-entrypoint "$NODE_CODE/grpo_two_phase.py" \
  --grpo-entrypoint-sha256 a7247e82f37f0e63708e7df0f4341849140db73c29453d6d8a11253bd2d6f354 \
  --frozen-shards "$FROZEN" --frozen-shards-sha256 "$FROZEN_SHA" \
  --generation-root "$GENERATION" --model-sums-sha256 "$MODEL_SUMS_SHA" \
  --compile-manifest "$COMPILE_MANIFEST/compile_manifest.jsonl" --compile-manifest-sha256 "$COMPILE_MANIFEST_SHA" \
  --receipt-origin compute_sft75344_lr2e4_minif2f --execution-domain slurm_node_local \
  --output "$COMPILE_OUTPUT" --lean-project "$PROJECT/lean_project" --task-spool-dir "$SPOOL" \
  --workers 4 --timeout 30 --warmup-timeout 3600 --queue-maxsize 256 --serial-prewarm

python3 - "$EVAL" "$MODEL_SUMS_SHA" "$FROZEN_SHA" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); summary=json.loads((root/'compile_persistent_v1/summary.json').read_text())
assert summary['status']=='complete' and summary['receipt_count']==7808 and summary['selected_attempt_count']==7808
assert summary['workers']==4 and summary['worker_lifecycle']=='persistent' and summary['prewarm_mode']=='serial'
assert not summary['fatal_errors']
generation=json.loads((root/'audit/GENERATION_AUDIT.json').read_text())
assert generation['status']=='PASS' and generation['problem_count']==244 and generation['attempt_count']==7808
payload={'schema':'minif2f_sft75344_lr2e4_pass32_execution_complete_v1','status':'PASS','problem_count':244,
 'attempt_count':7808,'split':'test','model_sums_sha256':sys.argv[2],'frozen_shards_sha256':sys.argv[3],
 'worker_lifecycle':'persistent','workers':4,'compile_timeout_seconds':30}
(root/'PERSISTENT_COMPILE_COMPLETE.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY

"$SFT_ENV/bin/python" "$NODE_CODE/analyze_corrected.py" --grpo-root "$EVAL" \
  --sft-eval-root "$BASELINE" --sft-receipt-root "$BASELINE" --output-dir "$EVAL/final_analysis"

python3 - "$EVAL" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); analysis=json.loads((root/'final_analysis/analysis.json').read_text())
assert analysis['status']=='PASS' and analysis['evaluation_scope']=={'split':'test','problem_count':244,'samples_per_problem':32}
r=analysis['grpo']; q=r['generation_quality']; c=r['compile_quality']
payload={'schema':'sft_main75344_lr2e4_minif2f_test_pass32_result_v1','status':'PASS','model':'deepseek-prover-v1.5-base+sft-main75344-r32-a64-lr2e4-epoch2',
 'pass_at_k_actual_prefix':r['pass_at_k_actual_prefix'],'pass_at_k_unbiased_estimator':r['pass_at_k_unbiased_estimator'],
 'attempt_success_rate':r['attempt_success_rate'],'length_truncation_count':q['hit_max_new_tokens_count'],
 'length_truncation_rate':q['hit_max_new_tokens_count']/7808,
 'strict_hallucinated_api_attempt_count':c['strict_hallucinated_api_attempt_count'],
 'strict_hallucinated_api_attempt_rate':c['strict_hallucinated_api_attempt_rate'],
 'analysis_path':str(root/'final_analysis/analysis.json')}
(root/'FINAL_RESULT.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
(root/'EXECUTION_COMPLETE.json').write_text(json.dumps({'status':'PASS','final_result':payload},indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY

#!/usr/bin/env bash
#SBATCH --job-name=grpo-dyn-108to144
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1/logs/dynamic108to144-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1/logs/dynamic108to144-%j.err

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1
COMMON=$ROOT/code/dynamic144_common.sh
source <(sed '/^# Cycle 0/,$d' "$COMMON")
SOURCE=/home/scc/bz22001004/runs/grpo-frozen8758-initialparams-20260901-v1/data/grpo_train8752_proof_free.jsonl
FULL_RUN_MAX_STEPS=144

python3 - "$ROOT" <<'PY'
import json,pathlib,sys
root=pathlib.Path(sys.argv[1]); checkpoint=root/'model/checkpoint-108'
marker=json.loads((root/'TRAINING_COMPLETE_108.json').read_text())
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert marker['status']=='PASS' and int(marker['global_step'])==108
assert int(state['global_step'])==108
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(4)]]:
    assert (checkpoint/name).is_file(), name
assert (root/'cycle-6/frozen/schedule/CYCLE_SCHEDULE_FROZEN.json').is_file()
print({'status':'PASS','resume_global_step':108})
PY

OLD_POLICY_SHA=$(python3 -c 'import json; print(json.load(open("'$ROOT'/cycle-6/frozen/CYCLE_FROZEN.json"))["old_policy_sha256"])')
ROOT="$ROOT" CYCLE_DIR="$ROOT/cycle-6/frozen" ADAPTER="$ROOT/model/checkpoint-108" \
  CYCLE_ID=dynamic48-cycle-6 OLD_POLICY_SHA256="$OLD_POLICY_SHA" \
  CYCLE_STOP_GLOBAL_STEP=112 FULL_MAX_STEPS=144 RESUME_CHECKPOINT="$ROOT/model/checkpoint-108" \
  bash "$ROOT/code/train_dynamic_cycle.template.sh"

prepare_fresh_cycle 7 "$ROOT/model/checkpoint-112" \
  "$ROOT/cycle-6/frozen/schedule/rollout_history.jsonl" 128 20260980
prepare_fresh_cycle 8 "$ROOT/model/checkpoint-128" \
  "$ROOT/cycle-7/frozen/schedule/rollout_history.jsonl" 144 20260990

python3 - "$ROOT" "$SOURCE" "$SLURM_JOB_ID" <<'PY'
import hashlib,json,pathlib,sys,time
root,source=map(pathlib.Path,sys.argv[1:3]); job=int(sys.argv[3]); checkpoint=root/'model/checkpoint-144'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert int(state['global_step'])==144
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(4)]]:
    assert (checkpoint/name).is_file(), name
payload={
  'schema':'dynamic_adaptive_grpo_144step_complete_v1','status':'PASS',
  'global_step':144,'slurm_job_id':job,'checkpoint':str(checkpoint),
  'adapter_model_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'total_16step_cycles':9,'total_512_problem_screens':9,
  'intermediate_evaluation_checkpoint':108,
  'continuation_source':str(source),
  'continuation_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'completed_at_unix':time.time(),
}
text=json.dumps(payload,indent=2,sort_keys=True)+'\n'
(root/'TRAINING_COMPLETE_144.json').write_text(text)
(root/'TRAINING_COMPLETE.json').write_text(text)
print(text,end='')
PY

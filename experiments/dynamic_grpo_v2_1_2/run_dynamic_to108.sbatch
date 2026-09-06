#!/usr/bin/env bash
#SBATCH --job-name=grpo-dyn-to108
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
#SBATCH --output=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1/logs/dynamic108-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1/logs/dynamic108-%j.err

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1
COMMON=$ROOT/code/dynamic144_common.sh
source <(sed '/^# Cycle 0/,$d' "$COMMON")
SOURCE=/home/scc/bz22001004/runs/grpo-frozen8758-initialparams-20260901-v1/data/grpo_train8752_proof_free.jsonl
FULL_RUN_MAX_STEPS=144

python3 - "$ROOT" "$SOURCE" <<'PY'
import json,pathlib,sys
root,source=map(pathlib.Path,sys.argv[1:])
checkpoint=root/'model/checkpoint-80'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert int(state['global_step'])==80
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(4)]]:
    assert (checkpoint/name).is_file(), name
rows=[json.loads(line) for line in source.open() if line.strip()]
ids={str(row['id']) for row in rows}
assert len(rows)==8752 and len(ids)==8752
history=[json.loads(line) for line in (root/'cycle-4/frozen/schedule/rollout_history.jsonl').open() if line.strip()]
seen={str(row['problem_id']) for row in history}
assert len(seen)==2560 and seen <= ids
assert len(ids-seen)>=4*512
print({'status':'PASS','global_step':80,'seen':len(seen),'remaining':len(ids-seen)})
PY

prepare_fresh_cycle 5 "$ROOT/model/checkpoint-80" \
  "$ROOT/cycle-4/frozen/schedule/rollout_history.jsonl" 96 20260960
prepare_fresh_cycle 6 "$ROOT/model/checkpoint-96" \
  "$ROOT/cycle-5/frozen/schedule/rollout_history.jsonl" 108 20260970

python3 - "$ROOT" "$SLURM_JOB_ID" <<'PY'
import hashlib,json,pathlib,sys,time
root=pathlib.Path(sys.argv[1]); job=int(sys.argv[2]); checkpoint=root/'model/checkpoint-108'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert int(state['global_step'])==108
for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt',*[f'rng_state_{i}.pth' for i in range(4)]]:
    assert (checkpoint/name).is_file(), name
payload={
  'schema':'dynamic_adaptive_grpo_108step_intermediate_v1','status':'PASS',
  'global_step':108,'slurm_job_id':job,'checkpoint':str(checkpoint),
  'adapter_model_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'evaluation_gate':'miniF2F-test-pass@32-before-resume-to-144',
  'completed_at_unix':time.time(),
}
(root/'TRAINING_COMPLETE_108.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY

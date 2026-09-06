#!/usr/bin/env bash
#SBATCH --job-name=grpo-dyn144-cont
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
#SBATCH --output=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1/logs/dynamic144-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1/logs/dynamic144-%j.err

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/grpo-dynamic48-two-new-screens-20260905-v1
COMMON=$ROOT/code/dynamic144_common.sh
FULL_SOURCE=/home/scc/bz22001004/runs/grpo-frozen8758-initialparams-20260901-v1/data/grpo_train8752_proof_free.jsonl

# Reuse the audited worker, rollout, compile, scheduling, and training functions
# without executing the original three-cycle driver body.
source <(sed '/^# Cycle 0/,$d' "$COMMON")
SOURCE=$FULL_SOURCE
FULL_RUN_MAX_STEPS=144

test -f "$ROOT/model/checkpoint-48/trainer_state.json"
test -f "$ROOT/TRAINING_COMPLETE.json"
python3 - "$ROOT" "$SOURCE" <<'PY'
import hashlib,json,pathlib,sys
root,source=map(pathlib.Path,sys.argv[1:])
state=json.loads((root/'model/checkpoint-48/trainer_state.json').read_text())
complete=json.loads((root/'TRAINING_COMPLETE.json').read_text())
assert int(state['global_step'])==48
assert complete['status']=='PASS' and int(complete['global_step'])==48
rows=[json.loads(line) for line in source.open() if line.strip()]
ids=[str(row['id']) for row in rows]
assert len(rows)==8752 and len(set(ids))==8752
assert all(row.get('schema_version')=='grpo_training_data_v2' for row in rows)
assert all(row.get('repeat')==1 for row in rows)
assert all(str(row['prompt']).endswith(':=\n') for row in rows)
history=[json.loads(line) for line in (root/'cycle-2/frozen/schedule/rollout_history.jsonl').open() if line.strip()]
seen={str(row['problem_id']) for row in history}
assert len(seen)==1536 and seen <= set(ids)
assert len(set(ids)-seen) >= 6*512
print(json.dumps({'status':'PASS','source_rows':len(rows),'seen_rows':len(seen),'remaining_unseen':len(set(ids)-seen)}))
PY

cp -n -- "$ROOT/TRAINING_COMPLETE.json" "$ROOT/TRAINING_COMPLETE_48.json"

for index in 3 4 5 6 7 8; do
  previous=$((index - 1))
  stop=$(((index + 1) * 16))
  seed=$((20260910 + index * 10))
  history=$ROOT/cycle-$previous/frozen/schedule/rollout_history.jsonl
  prepare_fresh_cycle "$index" "$ROOT/model/checkpoint-$((index * 16))" "$history" "$stop" "$seed"
done

test -f "$ROOT/model/checkpoint-144/trainer_state.json"
python3 - "$ROOT" "$SOURCE" "$SLURM_JOB_ID" <<'PY'
import hashlib,json,pathlib,sys,time
root,source=map(pathlib.Path,sys.argv[1:3]); job=int(sys.argv[3])
checkpoint=root/'model/checkpoint-144'
state=json.loads((checkpoint/'trainer_state.json').read_text())
assert int(state['global_step'])==144
payload={
  'schema':'dynamic_adaptive_grpo_144step_complete_v1',
  'status':'PASS','global_step':144,'slurm_job_id':job,
  'checkpoint':str(checkpoint),
  'adapter_model_sha256':hashlib.sha256((checkpoint/'adapter_model.safetensors').read_bytes()).hexdigest(),
  'total_16step_cycles':9,'total_512_problem_screens':9,
  'new_screen_cycles_after_reused_cycle0':8,
  'continuation_source':str(source),
  'continuation_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'completed_at_unix':time.time(),
}
text=json.dumps(payload,indent=2,sort_keys=True)+'\n'
(root/'TRAINING_COMPLETE_144.json').write_text(text)
(root/'TRAINING_COMPLETE.json').write_text(text)
print(text,end='')
PY

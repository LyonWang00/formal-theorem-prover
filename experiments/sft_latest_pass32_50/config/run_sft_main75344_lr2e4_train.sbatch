#!/usr/bin/env bash
#SBATCH --job-name=sft-main75344-r32-lr2e4
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/home/scc/bz22001004/runs/sft-main75344-deepseekprover-r32-lr2e4-epoch2-20260903-v1/logs/train-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/sft-main75344-deepseekprover-r32-lr2e4-epoch2-20260903-v1/logs/train-%j.err

set -euo pipefail

ROOT=/home/scc/bz22001004/runs/sft-main75344-deepseekprover-r32-lr2e4-epoch2-20260903-v1
PROJECT=/home/scc/bz22001004/projects/formal-theorem-prover
ENV=/home/scc/bz22001004/environments/formal-theorem-prover-sft
MODEL=$PROJECT/models/DeepSeek-Prover-V1.5-Base
DATASET=/home/scc/bz22001004/datasets/formal-theorem-prover/sft_main_75344_20260902_v1
TRAIN=$DATASET/train_data.jsonl
FULL_MANIFEST=$DATASET/train_manifest.jsonl
MANIFEST=$ROOT/input/train_manifest_hf_compatible_v2.jsonl
OUTPUT=$ROOT/model

mkdir -p "$ROOT/logs" "$ROOT/audit"
test ! -e "$OUTPUT"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 NCCL_ASYNC_ERROR_HANDLING=1

test "$(sha256sum "$TRAIN" | awk '{print $1}')" = 080c9426a57bea3ac3cf287ba1acb7900532eda66af8256387101c75cbcdefdf
test "$(sha256sum "$FULL_MANIFEST" | awk '{print $1}')" = cd8c2bea028672d3e6e6d902d844fbc0abb0c76efdf6b915e6a5c69b51ee3b93
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = e69bd07f416305c9a4f83289612f501a2030eac2ef812a3ba2b9c29d61228aa4
test "$(sha256sum "$DATASET/FROZEN.json" | awk '{print $1}')" = b3db8f140c599e10c0053d4df34a2d00e2db3cd59b991f6dd71701a30b22995e
test "$(sha256sum "$PROJECT/lean_prover/lean_training/sft.py" | awk '{print $1}')" = 80239da8dcefe3bc8f929f17ee7b1c25d0b17797e9787470ba3c6e690108266b
test "$(sha256sum "$PROJECT/lean_prover/lean_training/sft_pipeline/trainer.py" | awk '{print $1}')" = 4fa651039bf27906536948fb57dfb6b99fb4017ff6b02137232ae1a4edf3a34a
test "$(sha256sum "$MODEL/config.json" | awk '{print $1}')" = 9700b800d54caa843f1682982301e8e86ee26710b943d9c88ba9a7f257911ab5

python3 - "$TRAIN" "$MANIFEST" <<'PY'
import json,sys
data,manifest=sys.argv[1:]
with open(data,encoding='utf-8') as d, open(manifest,encoding='utf-8') as m:
    count=0
    for count,(dl,ml) in enumerate(zip(d,m),1):
        dr=json.loads(dl); mr=json.loads(ml)
        assert set(dr)=={'prompt','completion'}
        assert dr['prompt'].endswith(':= by sorry\n') and dr['completion'].startswith('by')
        assert 'repeat' not in dr
        assert mr['pantograph_verified'] is True and mr['verification_scope']=='full_proof'
    assert not d.readline() and not m.readline()
assert count==75344 and count%8==0
print(f'DATA_GATE PASS rows={count} global_batch_size=8 steps_per_epoch={count//8}')
PY

source "$ENV/bin/activate"
cd "$PROJECT"
nvidia-smi

"$ENV/bin/accelerate" launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  "$PROJECT/lean_prover/lean_training/sft.py" \
  --model_name_or_path "$MODEL" --train_file "$TRAIN" --train_manifest_file "$MANIFEST" --output_dir "$OUTPUT" \
  --sampling_strategy auto --require_pantograph_verified --require_supervised_eos \
  --max_seq_length 4096 --per_device_train_batch_size 1 --per_device_validation_batch_size 1 \
  --gradient_accumulation_steps 2 --dataloader_drop_last --dataloader_num_workers 4 \
  --max_steps -1 --learning_rate 2e-4 --num_train_epochs 2 --logging_steps 10 \
  --eval_strategy no --save_strategy steps --save_steps 300 --save_total_limit 4 \
  --warmup_ratio 0 --warmup_steps 0 --weight_decay 0.01 --lr_scheduler_type linear \
  --max_grad_norm 1 --gradient_checkpointing --optim adamw_torch --device_map local_rank \
  --lora_r 32 --lora_alpha 64 --lora_dropout 0.05 --seed 42 --data_seed 42

python3 - "$ROOT" "$SLURM_JOB_ID" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); job_id=int(sys.argv[2]); model=root/'model'
state=json.loads((model/'trainer_state.json').read_text())
config=json.loads((model/'training_config.json').read_text())['config']
adapter=json.loads((model/'adapter_config.json').read_text())
assert state['global_step']==18836 and abs(float(state['epoch'])-2.0)<1e-9, (state['global_step'],state['epoch'])
assert config['learning_rate']==2e-4 and config['lora_r']==32 and config['lora_alpha']==64
assert config['num_train_epochs']==2.0 and config['max_steps']==-1
assert adapter['r']==32 and adapter['lora_alpha']==64
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
payload={'schema':'sft_main75344_lr2e4_training_complete_v1','status':'PASS','job_id':job_id,
 'base_model':config['model_name_or_path'],'train_file':config['train_file'],'train_manifest_file':config['train_manifest_file'],
 'global_step':state['global_step'],'epoch':state['epoch'],'learning_rate':config['learning_rate'],
 'lora_r':config['lora_r'],'lora_alpha':config['lora_alpha'],'global_batch_size':8,
 'adapter_model_sha256':sha(model/'adapter_model.safetensors'),'trainer_state_sha256':sha(model/'trainer_state.json')}
(root/'TRAINING_COMPLETE.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
print(json.dumps(payload,indent=2,sort_keys=True))
PY

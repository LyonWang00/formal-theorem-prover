#!/usr/bin/env bash
# Run inside the long-lived 4-GPU allocation after the cycle schedule is frozen.
set -euo pipefail

: "${ROOT:?full dynamic run root required}"
: "${CYCLE_DIR:?cycle directory required}"
: "${ADAPTER:?cycle-start LoRA checkpoint required}"
: "${CYCLE_ID:?cycle ID required}"
: "${OLD_POLICY_SHA256:?merged old-policy SHA256SUMS hash required}"
: "${CYCLE_STOP_GLOBAL_STEP:?absolute stop step required}"
: "${PANTOGRAPH_POOL_SOCKET_DIR:?persistent Pantograph service required}"

FULL_MAX_STEPS=${FULL_MAX_STEPS:-149}

PROJECT=/home/scc/bz22001004/projects/formal-theorem-prover
ENV=/home/scc/bz22001004/environments/rollout-vllm-cu130
MODEL=$PROJECT/models/DeepSeek-Prover-V1.5-Base
CODE=$ROOT/code
TRAIN=$CYCLE_DIR/schedule/train_schedule.jsonl
MANIFEST=$CYCLE_DIR/schedule/CYCLE_SCHEDULE_FROZEN.json
OUTPUT=$ROOT/model
REWARD_LOG=$CYCLE_DIR/training_rewards.jsonl
RESUME_ARGS=()

test -f "$TRAIN"
test -f "$MANIFEST"
test "$(wc -l < "$TRAIN")" -eq 256
test "$FULL_MAX_STEPS" -gt 0
test "$CYCLE_STOP_GLOBAL_STEP" -le "$FULL_MAX_STEPS"
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  test -f "$RESUME_CHECKPOINT/trainer_state.json"
  RESUME_ARGS=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PANTOGRAPH_POOL_CLIENT_MODE=rank
export PATH="/home/scc/bz22001004/.elan/bin:$PATH"

"$ENV/bin/accelerate" launch --multi_gpu --num_processes 4 --num_machines 1 \
  --mixed_precision bf16 --dynamo_backend no \
  "$CODE/grpo_adaptive_remote.py" \
  --model_name_or_path "$MODEL" --adapter_path "$ADAPTER" \
  --train_file "$TRAIN" --output_dir "$OUTPUT" \
  --lean_project_path "$PROJECT/lean_project" --pantograph_imports Mathlib \
  --pantograph_num_workers 1 --pantograph_warmup_timeout 900 --lean_timeout 60 \
  --max_prompt_length 512 --max_completion_length 512 --num_generations 8 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 64 \
  --num_iterations 2 --num_train_epochs 100 --max_steps "$FULL_MAX_STEPS" \
  --learning_rate 1e-5 --logging_steps 1 --eval_strategy no \
  --save_strategy steps --save_steps 16 --save_total_limit 12 \
  --gradient_checkpointing --device_map local_rank --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --no-load_in_4bit \
  --use_vllm --vllm_mode colocate --vllm_enforce_eager --vllm_gpu_memory_utilization 0.48 \
  --vllm_tensor_parallel_size 1 --vllm_enable_sleep_mode \
  --compile_success_reward 1 --failure_reward 0 \
  --reward_log_file "$REWARD_LOG" \
  --adaptive_problems_per_update 16 \
  --adaptive_scale_ema_decay 0.90 \
  --adaptive_initial_scale 0.50 \
  --adaptive_trajectory_rank_beta 0.125 \
  --adaptive_cycle_schedule_manifest "$MANIFEST" \
  --adaptive_cycle_id "$CYCLE_ID" \
  --adaptive_old_policy_sha256 "$OLD_POLICY_SHA256" \
  --adaptive_cycle_stop_global_step "$CYCLE_STOP_GLOBAL_STEP" \
  --seed 20260909 --data_seed 20260909 \
  "${RESUME_ARGS[@]}"

test -f "$OUTPUT/checkpoint-$CYCLE_STOP_GLOBAL_STEP/trainer_state.json"
python3 - "$OUTPUT/checkpoint-$CYCLE_STOP_GLOBAL_STEP/trainer_state.json" "$CYCLE_STOP_GLOBAL_STEP" <<'PY'
import json, pathlib, sys
state=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert int(state['global_step']) == int(sys.argv[2]), state['global_step']
print(json.dumps({'status':'PASS','cycle_stop_global_step':int(sys.argv[2])}))
PY

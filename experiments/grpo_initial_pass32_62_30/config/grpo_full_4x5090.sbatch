#!/usr/bin/env bash
#SBATCH --job-name=grpo-e2-full
#SBATCH --partition=P107-RTX5090
#SBATCH --account=competition
#SBATCH --qos=qos_p107-rtx5090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:RTX5090:4
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/home/scc/bz22001004/runs/grpo-first5000-epoch2-20260829-v1/logs/full-%j.out
#SBATCH --error=/home/scc/bz22001004/runs/grpo-first5000-epoch2-20260829-v1/logs/full-%j.err

set -euo pipefail
ROOT=/home/scc/bz22001004/runs/grpo-first5000-epoch2-20260829-v1
PROJECT=/home/scc/bz22001004/projects/formal-theorem-prover
ENV=/home/scc/bz22001004/environments/rollout-vllm-cu130
MODEL=$PROJECT/models/DeepSeek-Prover-V1.5-Base
ADAPTER=/home/scc/bz22001004/runs/sft-resume-9344-to-18688-20260826/model/checkpoint-18688
TRAIN=${GRPO_TRAIN_FILE:-$ROOT/data/grpo_train5000_proof_free.jsonl}
OUTPUT=$ROOT/full/model
METRICS=$ROOT/full/gpu.csv

mkdir -p "$ROOT/logs" "$ROOT/full"
test ! -e "$OUTPUT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 TORCH_NCCL_ASYNC_ERROR_HANDLING=1 VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PATH="/home/scc/bz22001004/.elan/bin:$PATH"
source "$ROOT/stage_lean_olean_cache.sh"

nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw --format=csv > "$METRICS"
(
  while true; do
    printf '%s,' "$(date +%s)"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits | paste -sd ';' -
    sleep 5
  done
) >> "$METRICS" &
MONITOR_PID=$!
cleanup() { kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

cd "$PROJECT/lean_project"
lake env lean --version
cd "$PROJECT"
"$ENV/bin/accelerate" launch --multi_gpu --num_processes 4 --num_machines 1 --mixed_precision bf16 --dynamo_backend no \
  -m lean_prover.lean_training.grpo \
  --model_name_or_path "$MODEL" --adapter_path "$ADAPTER" \
  --train_file "$TRAIN" --output_dir "$OUTPUT" \
  --lean_project_path "$PROJECT/lean_project" --pantograph_imports Mathlib \
  --pantograph_num_workers 1 --pantograph_warmup_timeout 900 --lean_timeout 120 \
  --max_prompt_length 512 --max_completion_length 512 --num_generations 8 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 2 \
  --num_iterations 4 --num_train_epochs 1 --max_steps -1 \
  --learning_rate 1e-5 --logging_steps 1 --eval_strategy no \
  --save_strategy steps --save_steps 100 --save_total_limit 3 \
  --gradient_checkpointing --device_map local_rank --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
  --no-load_in_4bit \
  --use_vllm --vllm_mode colocate --vllm_enforce_eager --vllm_gpu_memory_utilization "${GRPO_VLLM_GPU_MEMORY_UTILIZATION:-0.48}" \
  --vllm_tensor_parallel_size 1 --vllm_enable_sleep_mode \
  --compile_success_reward 1 --failure_reward 0 \
  --reward_log_file "$ROOT/full/rewards.jsonl" \
  --seed 20260829 --data_seed 20260829

cleanup
trap - EXIT INT TERM
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,power.draw --format=csv

#!/usr/bin/env bash
set -euo pipefail

LINUX_PROJECT="${LINUX_PROJECT:-/home/lean/projects/formal-theorem-prover}"
PY="${PY:-$LINUX_PROJECT/.venv/bin/python}"
LEAN_PROJECT="${LEAN_PROJECT:-$LINUX_PROJECT/lean_project}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
ADAPTER_PATH="${ADAPTER_PATH:?ADAPTER_PATH is required}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PASS_K="${PASS_K:-2}"
NUM_BENCHMARK_SAMPLES="${NUM_BENCHMARK_SAMPLES:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
GENERATION_BACKEND="${GENERATION_BACKEND:-auto}"
GENERATOR_MAX_MODEL_LEN="${GENERATOR_MAX_MODEL_LEN:-4096}"
GENERATOR_GPU_MEMORY_UTILIZATION="${GENERATOR_GPU_MEMORY_UTILIZATION:-0.65}"
ENABLE_FLASHINFER_SAMPLER="${ENABLE_FLASHINFER_SAMPLER:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_NAME="${RUN_NAME:-benchmark_adapter}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
LOCAL_MINIF2F="${LOCAL_MINIF2F:-$LINUX_PROJECT/.cache/datasets/minif2f/test.jsonl}"

OUT_ROOT="$LINUX_PROJECT/outputs/${RUN_NAME}_${RUN_ID}"
BENCHMARK_FILE="$OUT_ROOT/benchmark${NUM_BENCHMARK_SAMPLES}.jsonl"
RESULTS_DIR="$OUT_ROOT/results"

export HF_ENDPOINT
source "$LINUX_PROJECT/scripts/lean_env_runtime.sh"

mkdir -p "$OUT_ROOT"
cd "$LINUX_PROJECT"

echo "LINUX_PROJECT=$LINUX_PROJECT"
echo "PY=$PY"
echo "LEAN_PROJECT=$LEAN_PROJECT"
echo "MODEL_NAME=$MODEL_NAME"
echo "ADAPTER_PATH=$ADAPTER_PATH"
echo "LOCAL_MINIF2F=$LOCAL_MINIF2F"
echo "NUM_WORKERS=$NUM_WORKERS"
echo "PASS_K=$PASS_K"
echo "NUM_BENCHMARK_SAMPLES=$NUM_BENCHMARK_SAMPLES"
echo "GENERATION_BACKEND=$GENERATION_BACKEND"
echo "GENERATOR_MAX_MODEL_LEN=$GENERATOR_MAX_MODEL_LEN"
echo "GENERATOR_GPU_MEMORY_UTILIZATION=$GENERATOR_GPU_MEMORY_UTILIZATION"
echo "ENABLE_FLASHINFER_SAMPLER=$ENABLE_FLASHINFER_SAMPLER"
echo "CUDA_HOME=${CUDA_HOME:-}"
echo "PATH=$PATH"
echo "OUT_ROOT=$OUT_ROOT"

if [ "$RUN_PREFLIGHT" = "1" ]; then
  echo
  echo "==== environment preflight ===="
  PREFLIGHT_ARGS=(
    "$LINUX_PROJECT/scripts/preflight_lean_env.py"
    --lean_project_path "$LEAN_PROJECT"
    --timeout 1200
    --output_json "$OUT_ROOT/preflight.json"
  )
  if [ "$ENABLE_FLASHINFER_SAMPLER" != "1" ]; then
    PREFLIGHT_ARGS+=(--skip_flashinfer_smoke)
  fi
  "$PY" "${PREFLIGHT_ARGS[@]}"
fi

echo
echo "==== prepare benchmark data ===="
echo "prepare_benchmark_start=$(date -Is)"
"$PY" -m lean_prover.lean_training.prepare_datasets \
  --benchmark_dataset_name "$LOCAL_MINIF2F" \
  --benchmark_sample_size "$NUM_BENCHMARK_SAMPLES" \
  --benchmark_output "$BENCHMARK_FILE" \
  --seed 20260711
echo "prepare_benchmark_end=$(date -Is)"

echo
echo "==== run adapter benchmark ===="
echo "benchmark_start=$(date -Is)"
/usr/bin/time -f "benchmark_elapsed_seconds=%e" "$PY" -m lean_prover.lean_training.benchmark_pipeline \
  --model_name_or_path "$MODEL_NAME" \
  --adapter_path "$ADAPTER_PATH" \
  --benchmark_file "$BENCHMARK_FILE" \
  --output_dir "$RESULTS_DIR" \
  --num_benchmark_samples "$NUM_BENCHMARK_SAMPLES" \
  --num_workers "$NUM_WORKERS" \
  --pass_k "$PASS_K" \
  --generation_backend "$GENERATION_BACKEND" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --vllm_max_model_len "$GENERATOR_MAX_MODEL_LEN" \
  --vllm_gpu_memory_utilization "$GENERATOR_GPU_MEMORY_UTILIZATION" \
  --temperature 0.7 \
  --top_p 0.95 \
  --load_in_4bit \
  --lean_project_path "$LEAN_PROJECT" \
  --imports Mathlib \
  --warmup_timeout 1200 \
  --lean_timeout 180 \
  --resume 2>&1 | tee "$OUT_ROOT/benchmark.log"
echo "benchmark_end=$(date -Is)"

echo
echo "==== result paths ===="
echo "OUT_ROOT=$OUT_ROOT"
echo "RESULTS_DIR=$RESULTS_DIR"
find "$RESULTS_DIR" -maxdepth 1 -type f | sort



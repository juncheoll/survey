#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODELS=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "huggyllama/llama-30b"
)

DEFAULT_COMPRESS_ONLY_MODELS=(
  "meta-llama/Llama-2-70b-hf"
)

MODELS=("${DEFAULT_MODELS[@]}")
if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
fi
COMPRESS_ONLY_MODELS=("${DEFAULT_COMPRESS_ONLY_MODELS[@]}")

MODEL_PATH="${MODEL_PATH:-_DUMMY_}"
GPU_BATCH_SIZES="${GPU_BATCH_SIZES:-1 2 4 8 16 32}"
COMPRESS_WEIGHT_MODES="${COMPRESS_WEIGHT_MODES:-off on}"
NUM_GPU_BATCHES="${NUM_GPU_BATCHES:-1}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
GEN_LEN="${GEN_LEN:-256}"
PERCENT_ARGS="${PERCENT_ARGS:-0 100 100 0 100 0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# If this script is run inside the documented Docker container, /logs is usually
# mounted from the host. Otherwise, keep logs inside the FlexGen directory.
if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/flexgen"
  else
    LOG_DIR="$SCRIPT_DIR/logs/flexgen"
  fi
fi

# shellcheck disable=SC2206
PERCENT=($PERCENT_ARGS)
if [[ "${#PERCENT[@]}" -ne 6 ]]; then
  echo "PERCENT_ARGS must contain exactly 6 numbers. Got: $PERCENT_ARGS" >&2
  exit 2
fi

# shellcheck disable=SC2206
BATCH_SIZES=($GPU_BATCH_SIZES)
if [[ "${#BATCH_SIZES[@]}" -eq 0 ]]; then
  echo "GPU_BATCH_SIZES must contain at least one number. Got: $GPU_BATCH_SIZES" >&2
  exit 2
fi

# shellcheck disable=SC2206
COMPRESS_MODES=($COMPRESS_WEIGHT_MODES)
if [[ "${#COMPRESS_MODES[@]}" -eq 0 ]]; then
  echo "COMPRESS_WEIGHT_MODES must contain at least one mode. Got: $COMPRESS_WEIGHT_MODES" >&2
  exit 2
fi

for compress_mode in "${COMPRESS_MODES[@]}"; do
  if [[ "$compress_mode" != "off" && "$compress_mode" != "on" ]]; then
    echo "COMPRESS_WEIGHT_MODES can only contain 'off' or 'on'. Got: $compress_mode" >&2
    exit 2
  fi
done

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

echo "[setup] working directory: $SCRIPT_DIR"
echo "[setup] log directory: $RUN_LOG_DIR"
echo "[setup] running uv sync"
if ! uv sync; then
  echo "[setup] uv sync failed" >&2
  exit 1
fi

VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[setup] virtualenv activation script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

# Activation is intentional here because FlexGen's invocation uses python -m.
# This also matches the manual workflow in README.md.
# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

{
  echo "# framework: FlexGen"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# model_path: $MODEL_PATH"
  echo "# percent: ${PERCENT[*]}"
  echo "# gpu_batch_sizes: ${BATCH_SIZES[*]}"
  echo "# compress_weight_modes: ${COMPRESS_MODES[*]}"
  echo "# compress_only_models: ${COMPRESS_ONLY_MODELS[*]}"
  echo "# num_gpu_batches: $NUM_GPU_BATCHES"
  echo "# prompt_len: $PROMPT_LEN"
  echo "# gen_len: $GEN_LEN"
  printf "started_at\tended_at\tduration_sec\tmodel\tgpu_batch_size\tcompress_weight\tstatus\texit_code\tstdout_log\tmetrics_log\n"
} > "$SUMMARY_LOG"

total=0
success=0
failed=0

run_one() {
  local model="$1"
  local gpu_batch_size="$2"
  local compress_mode="$3"
  local safe_model
  local quant_label
  local stdout_log
  local metrics_log
  local started_at
  local started_sec
  local ended_at
  local ended_sec
  local duration_sec
  local exit_code
  local status
  local -a compress_args
  local -a cmd

  total=$((total + 1))

  safe_model="${model//\//_}"
  safe_model="${safe_model//:/_}"
  if [[ "$compress_mode" == "on" ]]; then
      quant_label="compress_weight"
      compress_args=(--compress-weight)
  else
    quant_label="no_compress"
    compress_args=()
  fi
  stdout_log="$RUN_LOG_DIR/${safe_model}_gbs${gpu_batch_size}_${quant_label}.stdout.log"
  metrics_log="$RUN_LOG_DIR/${safe_model}_gbs${gpu_batch_size}_${quant_label}.metrics.log"

  cmd=(
    "$PYTHON_BIN" -m flexllmgen.flex_opt
    --model "$model"
    --percent "${PERCENT[@]}"
    --path "$MODEL_PATH"
    --gpu-batch-size "$gpu_batch_size"
    --num-gpu-batches "$NUM_GPU_BATCHES"
    --prompt-len "$PROMPT_LEN"
    --gen-len "$GEN_LEN"
    --log-file "$metrics_log"
    "${compress_args[@]}"
  )

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] $model gbs=$gpu_batch_size compress_weight=$compress_mode"
  {
    echo "# started_at: $started_at"
    printf "# command:"
    printf " %q" "${cmd[@]}"
    echo
    echo
    "${cmd[@]}"
  } > "$stdout_log" 2>&1
  exit_code=$?

  ended_at="$(date -Iseconds)"
  ended_sec="$(date +%s)"
  duration_sec=$((ended_sec - started_sec))

  if [[ "$exit_code" -eq 0 ]]; then
    status="success"
    success=$((success + 1))
    echo "[run] success: $model gbs=$gpu_batch_size compress_weight=$compress_mode (${duration_sec}s)"
  else
    status="failed"
    failed=$((failed + 1))
    echo "[run] failed: $model gbs=$gpu_batch_size compress_weight=$compress_mode (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$model" "$gpu_batch_size" "$compress_mode" "$status" "$exit_code" \
    "$stdout_log" "$metrics_log" >> "$SUMMARY_LOG"
}

for model in "${MODELS[@]}"; do
  for gpu_batch_size in "${BATCH_SIZES[@]}"; do
    for compress_mode in "${COMPRESS_MODES[@]}"; do
      run_one "$model" "$gpu_batch_size" "$compress_mode"
    done
  done
done

for model in "${COMPRESS_ONLY_MODELS[@]}"; do
  for gpu_batch_size in "${BATCH_SIZES[@]}"; do
    for compress_mode in "${COMPRESS_MODES[@]}"; do
      if [[ "$compress_mode" == "on" ]]; then
        run_one "$model" "$gpu_batch_size" "$compress_mode"
      fi
    done
  done
done

{
  echo
  echo "# completed_at: $(date -Iseconds)"
  echo "# total: $total"
  echo "# success: $success"
  echo "# failed: $failed"
} >> "$SUMMARY_LOG"

echo "[done] total=$total success=$success failed=$failed"
echo "[done] summary: $SUMMARY_LOG"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

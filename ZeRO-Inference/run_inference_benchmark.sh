#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/DeepSpeedExamples/inference/huggingface/zero_inference"

cd "$SCRIPT_DIR"

DEFAULT_MODELS=(
  "meta-llama/Llama-2-7b-hf"
  "meta-llama/Llama-2-13b-hf"
  "huggyllama/llama-30b"
)

MODELS=("${DEFAULT_MODELS[@]}")
if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
fi

GPU_BATCH_SIZES="${GPU_BATCH_SIZES:-1 2 4 8}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
GEN_LEN="${GEN_LEN:-512}"
NUM_GPUS="${NUM_GPUS:-1}"
DEEPSPEED_BIN="${DEEPSPEED_BIN:-deepspeed}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/zero_inference"
  else
    LOG_DIR="$SCRIPT_DIR/logs/zero_inference"
  fi
fi

# shellcheck disable=SC2206
BATCH_SIZES=($GPU_BATCH_SIZES)
if [[ "${#BATCH_SIZES[@]}" -eq 0 ]]; then
  echo "GPU_BATCH_SIZES must contain at least one number. Got: $GPU_BATCH_SIZES" >&2
  exit 2
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

echo "[setup] working directory: $SCRIPT_DIR"
echo "[setup] benchmark directory: $BENCH_DIR"
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

# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

if [[ ! -d "$BENCH_DIR" ]]; then
  echo "[setup] benchmark directory not found: $BENCH_DIR" >&2
  exit 1
fi

cd "$BENCH_DIR"

{
  echo "# framework: ZeRO-Inference"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# num_gpus: $NUM_GPUS"
  echo "# gpu_batch_sizes: ${BATCH_SIZES[*]}"
  echo "# prompt_len: $PROMPT_LEN"
  echo "# gen_len: $GEN_LEN"
  echo "# cpu_offload: true"
  echo "# kv_offload: true"
  echo "# dummy: true"
  printf "started_at\tended_at\tduration_sec\tmodel\tbatch_size\tstatus\texit_code\tstdout_log\tmetrics_log\n"
} > "$SUMMARY_LOG"

total=0
success=0
failed=0

for model in "${MODELS[@]}"; do
  for batch_size in "${BATCH_SIZES[@]}"; do
    total=$((total + 1))

    safe_model="${model//\//_}"
    safe_model="${safe_model//:/_}"
    stdout_log="$RUN_LOG_DIR/${safe_model}_bs${batch_size}.stdout.log"
    metrics_log="$RUN_LOG_DIR/${safe_model}_bs${batch_size}.metrics.log"

    cmd=(
      "$DEEPSPEED_BIN" --num_gpus "$NUM_GPUS" run_model.py
      --dummy
      --model "$model"
      --batch-size "$batch_size"
      --prompt-len "$PROMPT_LEN"
      --gen-len "$GEN_LEN"
      --cpu-offload
      --kv-offload
      --log-file "$metrics_log"
    )

    started_at="$(date -Iseconds)"
    started_sec="$(date +%s)"

    echo "[run] $model bs=$batch_size"
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
      echo "[run] success: $model bs=$batch_size (${duration_sec}s)"
    else
      status="failed"
      failed=$((failed + 1))
      echo "[run] failed: $model bs=$batch_size (${duration_sec}s, exit_code=$exit_code)"
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "$duration_sec" "$model" "$batch_size" "$status" "$exit_code" \
      "$stdout_log" "$metrics_log" >> "$SUMMARY_LOG"
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

#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODELS=(
  "meta-llama/Llama-2-70b-hf"
)

DEFAULT_COMPRESS_ONLY_MODELS=(
  "meta-llama/Llama-2-70b-hf"
)

MODELS=("${DEFAULT_MODELS[@]}")
if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
fi
COMPRESS_ONLY_MODELS=("${DEFAULT_COMPRESS_ONLY_MODELS[@]}")

HEAD_IP="${HEAD_IP:-$(hostname -i | awk '{print $1}')}"
PORT="${PORT:-7777}"
REMOTE_FLEXGEN_DIR="${REMOTE_FLEXGEN_DIR:-$SCRIPT_DIR}"
PYTHON_EXEC="${PYTHON_EXEC:-$REMOTE_FLEXGEN_DIR/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-_DUMMY_}"
GPU_BATCH_SIZES="${GPU_BATCH_SIZES:-1 2 4 8 16 32 64 128 256}"
COMPRESS_WEIGHT_MODES="${COMPRESS_WEIGHT_MODES:-off on}"
NUM_GPU_BATCHES="${NUM_GPU_BATCHES:-1}"
PROMPT_LEN="${PROMPT_LEN:-1024}"
GEN_LEN="${GEN_LEN:-512}"
PERCENT_ARGS="${PERCENT_ARGS:-80 20 100 0 100 0}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
CORES_PER_GPU="${CORES_PER_GPU:-4}"
RUN_SETUP="${RUN_SETUP:-1}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"
MPI_PREFIX="${MPI_PREFIX:-}"
SSH_PORT="${SSH_PORT:-}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/flexgen_dist"
  else
    LOG_DIR="$SCRIPT_DIR/logs/flexgen_dist"
  fi
fi

MPI_HOST_ARGS=()
HOSTS_LABEL=""

if [[ -n "${HOSTFILE:-}" ]]; then
  MPI_HOST_ARGS=(--hostfile "$HOSTFILE")
  HOSTS_LABEL="hostfile:$HOSTFILE"
elif [[ -n "${HOSTS:-}" ]]; then
  MPI_HOST_ARGS=(-H "$HOSTS")
  HOSTS_LABEL="$HOSTS"
else
  HOSTS="$HEAD_IP"
  MPI_HOST_ARGS=(-H "$HOSTS")
  HOSTS_LABEL="$HOSTS"
fi

# shellcheck disable=SC2206
MPI_EXTRA=($MPI_EXTRA_ARGS)
if [[ -n "$MPI_PREFIX" ]]; then
  MPI_EXTRA+=(--prefix "$MPI_PREFIX")
fi
if [[ -n "$SSH_PORT" ]]; then
  MPI_EXTRA+=(--mca plm_rsh_args "-p $SSH_PORT")
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

if [[ "$RUN_SETUP" != "0" ]]; then
  echo "[setup] validating remote uv environments"
  "$SCRIPT_DIR/run_flexgen_dist_setup.sh"
fi

{
  echo "# framework: FlexGen distributed"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# hosts: $HOSTS_LABEL"
  echo "# head_ip: $HEAD_IP"
  echo "# port: $PORT"
  echo "# remote_flexgen_dir: $REMOTE_FLEXGEN_DIR"
  echo "# python_exec: $PYTHON_EXEC"
  echo "# model_path: $MODEL_PATH"
  echo "# percent: ${PERCENT[*]}"
  echo "# gpus_per_node: $GPUS_PER_NODE"
  echo "# cores_per_gpu: $CORES_PER_GPU"
  echo "# mpi_prefix: ${MPI_PREFIX:-default}"
  echo "# ssh_port: ${SSH_PORT:-default}"
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
    mpirun
    "${MPI_HOST_ARGS[@]}"
    "${MPI_EXTRA[@]}"
    --mca btl_tcp_if_exclude lo,docker0
    --mca oob_tcp_if_exclude lo,docker0
    --map-by "ppr:${GPUS_PER_NODE}:node:pe=${CORES_PER_GPU}"
    --oversubscribe
    --bind-to core
    -x "OMP_NUM_THREADS=${CORES_PER_GPU}"
    -x PATH
    -x HF_HOME
    -x HF_TOKEN
    -x HUGGING_FACE_HUB_TOKEN
    -x TRANSFORMERS_CACHE
    "$PYTHON_EXEC" -m flexllmgen.dist_flex_opt
    --head-ip "$HEAD_IP"
    --port "$PORT"
    --use-mpi
    --model "$model"
    --path "$MODEL_PATH"
    --gpu-batch-size "$gpu_batch_size"
    --num-gpu-batches "$NUM_GPU_BATCHES"
    --prompt-len "$PROMPT_LEN"
    --gen-len "$GEN_LEN"
    --percent "${PERCENT[@]}"
    --comm-device gpu
    --async-comm
    --no-log
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

  grep -E "model size:|peak gpu mem:|prefill latency:|decode latency:|total latency:" \
    "$stdout_log" > "$metrics_log" || true

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

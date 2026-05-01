#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-meta-llama/Llama-2-70b-hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
REMOTE_VLLM_DIR="${REMOTE_VLLM_DIR:-$SCRIPT_DIR}"
RAY_PORT="${RAY_PORT:-6379}"
SERVE_HOST="${SERVE_HOST:-0.0.0.0}"
SERVE_PORT="${SERVE_PORT:-8000}"
BENCH_HOST="${BENCH_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-auto}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-auto}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
RUN_RAY_SETUP="${RUN_RAY_SETUP:-1}"
STOP_RAY_ON_EXIT="${STOP_RAY_ON_EXIT:-0}"
VLLM_DISTRIBUTED_EXECUTOR_BACKEND="${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-auto}"
SERVER_START_TIMEOUT_SEC="${SERVER_START_TIMEOUT_SEC:-900}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-512}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCIES="${MAX_CONCURRENCIES:-1 2 4 8 16 32 64 128 256}"
VLLM_EXTRA_SERVE_ARGS="${VLLM_EXTRA_SERVE_ARGS:-}"
VLLM_EXTRA_BENCH_ARGS="${VLLM_EXTRA_BENCH_ARGS:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"
IGNORE_EOS="${IGNORE_EOS:-1}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/vllm"
  else
    LOG_DIR="$SCRIPT_DIR/logs/vllm"
  fi
fi

HOST_NAMES=()
HOST_SLOTS=()
TOTAL_GPUS=0

add_host() {
  local host="$1"
  local slots="$2"
  HOST_NAMES+=("$host")
  HOST_SLOTS+=("$slots")
  TOTAL_GPUS=$((TOTAL_GPUS + slots))
}

parse_hostfile() {
  local line
  local host
  local slots
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -z "$line" ]] && continue

    host="$(awk '{print $1}' <<< "$line")"
    slots="$(grep -oE 'slots=[0-9]+' <<< "$line" | head -1 | cut -d= -f2)"
    slots="${slots:-$GPUS_PER_NODE}"
    add_host "$host" "$slots"
  done < "$1"
}

if [[ -n "${HOSTFILE:-}" ]]; then
  parse_hostfile "$HOSTFILE"
elif [[ -n "${HOSTS:-}" ]]; then
  IFS=',' read -r -a HOST_LIST <<< "$HOSTS"
  for host in "${HOST_LIST[@]}"; do
    add_host "$host" "$GPUS_PER_NODE"
  done
else
  add_host "127.0.0.1" "$GPUS_PER_NODE"
fi

if [[ "$TOTAL_GPUS" -eq 0 ]]; then
  echo "No GPU slots found. Set HOSTFILE or HOSTS." >&2
  exit 2
fi

HEAD_HOST="${HEAD_HOST:-${HOST_NAMES[0]}}"
HEAD_ADDRESS="${HEAD_ADDRESS:-$(hostname -i | awk '{print $1}')}"

if [[ "$TENSOR_PARALLEL_SIZE" == "auto" ]]; then
  TENSOR_PARALLEL_SIZE="${HOST_SLOTS[0]}"
fi

if [[ "$PIPELINE_PARALLEL_SIZE" == "auto" ]]; then
  PIPELINE_PARALLEL_SIZE="${#HOST_NAMES[@]}"
fi

PARALLEL_WORLD_SIZE=$((TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE))
if [[ "$PARALLEL_WORLD_SIZE" -ne "$TOTAL_GPUS" ]]; then
  echo "Warning: TP x PP = $PARALLEL_WORLD_SIZE, but host slots total = $TOTAL_GPUS" >&2
fi

if [[ "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" == "auto" ]]; then
  if [[ "${#HOST_NAMES[@]}" -gt 1 ]]; then
    VLLM_DISTRIBUTED_EXECUTOR_BACKEND="ray"
  else
    VLLM_DISTRIBUTED_EXECUTOR_BACKEND=""
  fi
fi

# shellcheck disable=SC2206
CONCURRENCY_LIST=($MAX_CONCURRENCIES)
if [[ "${#CONCURRENCY_LIST[@]}" -eq 0 ]]; then
  echo "MAX_CONCURRENCIES must contain at least one number. Got: $MAX_CONCURRENCIES" >&2
  exit 2
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
SERVER_LOG="$RUN_LOG_DIR/server.stdout.log"
mkdir -p "$RUN_LOG_DIR"

cleanup() {
  local exit_code=$?
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[cleanup] stopping vLLM server pid=$SERVER_PID"
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$STOP_RAY_ON_EXIT" != "0" ]]; then
    echo "[cleanup] stopping Ray"
    ray stop --force >/dev/null 2>&1 || true
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

echo "[setup] working directory: $SCRIPT_DIR"
echo "[setup] log directory: $RUN_LOG_DIR"
echo "[setup] hosts: ${HOST_NAMES[*]}"
echo "[setup] total gpu slots: $TOTAL_GPUS"

echo "[setup] running uv sync on head"
uv sync
source "$SCRIPT_DIR/.venv/bin/activate"

if [[ "$RUN_RAY_SETUP" != "0" && "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" == "ray" ]]; then
  "$SCRIPT_DIR/run_vllm_ray_setup.sh"
fi

{
  echo "# framework: vLLM"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# model: $MODEL"
  echo "# hosts: ${HOST_NAMES[*]}"
  echo "# slots: ${HOST_SLOTS[*]}"
  echo "# total_gpus: $TOTAL_GPUS"
  echo "# tensor_parallel_size: $TENSOR_PARALLEL_SIZE"
  echo "# pipeline_parallel_size: $PIPELINE_PARALLEL_SIZE"
  echo "# parallel_world_size: $PARALLEL_WORLD_SIZE"
  echo "# distributed_executor_backend: ${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-vllm_default}"
  echo "# max_model_len: $MAX_MODEL_LEN"
  echo "# max_num_seqs: ${MAX_NUM_SEQS:-vllm_default}"
  echo "# gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
  echo "# enforce_eager: $VLLM_ENFORCE_EAGER"
  echo "# disable_custom_all_reduce: $VLLM_DISABLE_CUSTOM_ALL_REDUCE"
  echo "# random_input_len: $RANDOM_INPUT_LEN"
  echo "# random_output_len: $RANDOM_OUTPUT_LEN"
  echo "# num_prompts: $NUM_PROMPTS"
  echo "# max_concurrencies: ${CONCURRENCY_LIST[*]}"
  printf "started_at\tended_at\tduration_sec\tmodel\tmax_concurrency\tstatus\texit_code\tserver_log\tbench_log\tresult_json\n"
} > "$SUMMARY_LOG"

# shellcheck disable=SC2206
SERVE_EXTRA=($VLLM_EXTRA_SERVE_ARGS)
# shellcheck disable=SC2206
BENCH_EXTRA=($VLLM_EXTRA_BENCH_ARGS)

serve_cmd=(
  vllm serve "$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
  --pipeline-parallel-size "$PIPELINE_PARALLEL_SIZE"
  --host "$SERVE_HOST"
  --port "$SERVE_PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
)

if [[ -n "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
  serve_cmd+=(--distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
fi

if [[ -n "$MAX_NUM_SEQS" ]]; then
  serve_cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
fi

if [[ "$VLLM_ENFORCE_EAGER" != "0" ]]; then
  serve_cmd+=(--enforce-eager)
fi

if [[ "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" != "0" ]]; then
  serve_cmd+=(--disable-custom-all-reduce)
fi

serve_cmd+=("${SERVE_EXTRA[@]}")

echo "[serve] starting vLLM server"
{
  printf "# command:"
  printf " %q" "${serve_cmd[@]}"
  echo
  echo
  "${serve_cmd[@]}"
} > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo "[serve] waiting for http://$BENCH_HOST:$SERVE_PORT/v1/models"
deadline=$((SECONDS + SERVER_START_TIMEOUT_SEC))
until "$SCRIPT_DIR/.venv/bin/python" - "$BENCH_HOST" "$SERVE_PORT" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2) as response:
        sys.exit(0 if response.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] server exited before becoming ready. See $SERVER_LOG" >&2
    exit 1
  fi
  if [[ "$SECONDS" -ge "$deadline" ]]; then
    echo "[serve] timed out waiting for server. See $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done

failed=0

for max_concurrency in "${CONCURRENCY_LIST[@]}"; do
  bench_log="$RUN_LOG_DIR/bench_concurrency_${max_concurrency}.stdout.log"
  result_json="$RUN_LOG_DIR/bench_concurrency_${max_concurrency}.json"

  bench_cmd=(
    vllm bench serve
    --backend openai
    --model "$SERVED_MODEL_NAME"
    --base-url "http://$BENCH_HOST:$SERVE_PORT"
    --endpoint /v1/completions
    --dataset-name random
    --num-prompts "$NUM_PROMPTS"
    --random-input-len "$RANDOM_INPUT_LEN"
    --random-output-len "$RANDOM_OUTPUT_LEN"
    --random-range-ratio 0
    --request-rate "$REQUEST_RATE"
    --max-concurrency "$max_concurrency"
    --save-result
    --result-dir "$RUN_LOG_DIR"
    --result-filename "$(basename "$result_json")"
    "${BENCH_EXTRA[@]}"
  )

  if [[ "$IGNORE_EOS" != "0" ]]; then
    bench_cmd+=(--ignore-eos)
  fi

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"
  echo "[bench] running benchmark max_concurrency=$max_concurrency"
  {
    printf "# command:"
    printf " %q" "${bench_cmd[@]}"
    echo
    echo
    "${bench_cmd[@]}"
  } > "$bench_log" 2>&1
  exit_code=$?
  ended_at="$(date -Iseconds)"
  ended_sec="$(date +%s)"
  duration_sec=$((ended_sec - started_sec))

  if [[ "$exit_code" -eq 0 ]]; then
    status="success"
    echo "[bench] success max_concurrency=$max_concurrency (${duration_sec}s)"
  else
    status="failed"
    failed=$((failed + 1))
    echo "[bench] failed max_concurrency=$max_concurrency (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$MODEL" "$max_concurrency" "$status" "$exit_code" \
    "$SERVER_LOG" "$bench_log" "$result_json" >> "$SUMMARY_LOG"
done

echo "[done] summary: $SUMMARY_LOG"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

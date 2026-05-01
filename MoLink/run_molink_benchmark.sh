#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${MODEL:-meta-llama/Llama-2-70b-hf}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL}"
MODEL_LAYERS="${MODEL_LAYERS:-80}"
REMOTE_MOLINK_DIR="${REMOTE_MOLINK_DIR:-$SCRIPT_DIR}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
UV_BIN="${UV_BIN:-uv}"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
VLLM_BIN="${VLLM_BIN:-.venv/bin/vllm}"
SSH_PORT="${SSH_PORT:-}"
SSH_EXTRA_ARGS="${SSH_EXTRA_ARGS:-}"

HEAD_ADDRESS="${HEAD_ADDRESS:-$(hostname -i | awk '{print $1}')}"
API_PORT="${API_PORT:-8080}"
WORKER_API_PORT_BASE="${WORKER_API_PORT_BASE:-9095}"
GRPC_PORT_BASE="${GRPC_PORT_BASE:-50061}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MOLINK_ENFORCE_EAGER="${MOLINK_ENFORCE_EAGER:-1}"
SERVER_START_TIMEOUT_SEC="${SERVER_START_TIMEOUT_SEC:-900}"
STAGE_START_TIMEOUT_SEC="${STAGE_START_TIMEOUT_SEC:-$SERVER_START_TIMEOUT_SEC}"
SERVER_START_STAGGER_SEC="${SERVER_START_STAGGER_SEC:-5}"
PIPELINE_READY_GRACE_SEC="${PIPELINE_READY_GRACE_SEC:-5}"
CLEANUP_EXISTING_SERVERS="${CLEANUP_EXISTING_SERVERS:-1}"

NUM_PROMPTS="${NUM_PROMPTS:-64}"
NUM_PROMPTS_PER_CONCURRENCY="${NUM_PROMPTS_PER_CONCURRENCY:-0}"
RANDOM_INPUT_LEN="${RANDOM_INPUT_LEN:-1024}"
RANDOM_OUTPUT_LEN="${RANDOM_OUTPUT_LEN:-256}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
MAX_CONCURRENCIES="${MAX_CONCURRENCIES:-1 2 4 16 32 64 128 256}"
IGNORE_EOS="${IGNORE_EOS:-1}"
MOLINK_EXTRA_SERVER_ARGS="${MOLINK_EXTRA_SERVER_ARGS:-}"
VLLM_EXTRA_BENCH_ARGS="${VLLM_EXTRA_BENCH_ARGS:-}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/molink"
  else
    LOG_DIR="$SCRIPT_DIR/logs/molink"
  fi
fi

SETUP_HOSTS=()
STAGE_HOSTS=()
STAGE_GPU_IDS=()
STAGE_LABELS=()

has_setup_host() {
  local candidate="$1"
  local existing
  for existing in "${SETUP_HOSTS[@]:-}"; do
    if [[ "$existing" == "$candidate" ]]; then
      return 0
    fi
  done
  return 1
}

add_host() {
  local host="$1"
  local slots="$2"
  local gpu

  if ! [[ "$slots" =~ ^[0-9]+$ ]] || [[ "$slots" -lt 1 ]]; then
    echo "MoLink host slots must be a positive integer. Host $host has slots=$slots." >&2
    exit 2
  fi

  if ! has_setup_host "$host"; then
    SETUP_HOSTS+=("$host")
  fi

  for ((gpu = 0; gpu < slots; gpu++)); do
    STAGE_HOSTS+=("$host")
    STAGE_GPU_IDS+=("$gpu")
    STAGE_LABELS+=("${host}:gpu${gpu}")
  done
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

NUM_STAGES="${#STAGE_HOSTS[@]}"
if [[ "$NUM_STAGES" -eq 0 ]]; then
  echo "No hosts found. Set HOSTFILE or HOSTS." >&2
  exit 2
fi

HEAD_HOST="${HEAD_HOST:-${STAGE_HOSTS[0]}}"

if [[ "$MODEL_LAYERS" -lt "$NUM_STAGES" ]]; then
  echo "MODEL_LAYERS=$MODEL_LAYERS is smaller than number of stages=$NUM_STAGES." >&2
  exit 2
fi

# shellcheck disable=SC2206
CONCURRENCY_LIST=($MAX_CONCURRENCIES)
if [[ "${#CONCURRENCY_LIST[@]}" -eq 0 ]]; then
  echo "MAX_CONCURRENCIES must contain at least one number. Got: $MAX_CONCURRENCIES" >&2
  exit 2
fi

SSH_ARGS=()
# shellcheck disable=SC2206
SSH_ARGS+=($SSH_EXTRA_ARGS)
if [[ -n "$SSH_PORT" ]]; then
  SSH_ARGS+=(-p "$SSH_PORT")
fi

remote_quote() {
  printf "%q" "$1"
}

run_remote() {
  local host="$1"
  local command="$2"
  ssh "${SSH_ARGS[@]}" "$host" "bash -lc $(remote_quote "$command")"
}

check_local_http_models() {
  local host="$1"
  local port="$2"
  "$SCRIPT_DIR/.venv/bin/python" - "$host" "$port" <<'PY'
import sys
import urllib.request

host, port = sys.argv[1], sys.argv[2]
try:
    with urllib.request.urlopen(f"http://{host}:{port}/v1/models", timeout=2) as response:
        sys.exit(0 if response.status < 500 else 1)
except Exception:
    sys.exit(1)
PY
}

check_local_tcp() {
  local host="$1"
  local port="$2"
  "$SCRIPT_DIR/.venv/bin/python" - "$host" "$port" <<'PY'
import socket
import sys

host, port = sys.argv[1], int(sys.argv[2])
try:
    with socket.create_connection((host, port), timeout=2):
        sys.exit(0)
except OSError:
    sys.exit(1)
PY
}

check_remote_tcp() {
  local host="$1"
  local port="$2"
  local remote_dir
  local command
  remote_dir="$(remote_quote "$REMOTE_MOLINK_DIR")"
  command="cd $remote_dir && .venv/bin/python -c 'import socket, sys; socket.create_connection((\"127.0.0.1\", int(sys.argv[1])), timeout=2).close()' $(remote_quote "$port")"
  run_remote "$host" "$command" >/dev/null 2>&1
}

check_stage_grpc() {
  local host="$1"
  local port="$2"
  if [[ "$host" == "$HEAD_HOST" ]]; then
    check_local_tcp "127.0.0.1" "$port" || check_local_tcp "$HEAD_ADDRESS" "$port"
  else
    check_remote_tcp "$host" "$port"
  fi
}

check_head_topology_count() {
  local expected="$1"
  "$SCRIPT_DIR/.venv/bin/python" - "$HEAD_PEER" "$expected" <<'PY'
import asyncio
import sys

import grpc

from molinkv1.comm import molink_pb2, molink_pb2_grpc
from molinkv1.utils import get_grpc_options


async def main() -> int:
    address, expected = sys.argv[1], int(sys.argv[2])
    channel = grpc.aio.insecure_channel(address, options=get_grpc_options())
    try:
        stub = molink_pb2_grpc.MolinkServiceStub(channel)
        topology = await stub.GetTopology(molink_pb2.HealthCheckRequest(), timeout=2)
        return 0 if len(topology.nodes) >= expected else 1
    except Exception:
        return 1
    finally:
        await channel.close()


raise SystemExit(asyncio.run(main()))
PY
}

wait_for_stage_grpc() {
  local idx="$1"
  local host="$2"
  local port="$3"
  local pid="$4"
  local log_file="$5"
  local deadline

  echo "[serve] waiting for stage=$idx host=$host grpc_port=$port"
  deadline=$((SECONDS + STAGE_START_TIMEOUT_SEC))
  until check_stage_grpc "$host" "$port"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] server process exited during startup: stage=$idx host=$host grpc_port=$port" >&2
      echo "[serve] log: $log_file" >&2
      tail -n 80 "$log_file" >&2 || true
      exit 1
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "[serve] timed out waiting for stage=$idx host=$host grpc_port=$port" >&2
      echo "[serve] log: $log_file" >&2
      tail -n 80 "$log_file" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "[serve] stage=$idx grpc is ready"
}

wait_for_topology_count() {
  local expected="$1"
  local idx="$2"
  local pid="$3"
  local log_file="$4"
  local deadline

  echo "[serve] waiting for head topology to include $expected stage(s)"
  deadline=$((SECONDS + STAGE_START_TIMEOUT_SEC))
  until check_head_topology_count "$expected"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] server process exited before joining topology: stage=$idx" >&2
      echo "[serve] log: $log_file" >&2
      tail -n 80 "$log_file" >&2 || true
      exit 1
    fi
    if [[ "$SECONDS" -ge "$deadline" ]]; then
      echo "[serve] timed out waiting for stage=$idx to join head topology" >&2
      echo "[serve] log: $log_file" >&2
      tail -n 80 "$log_file" >&2 || true
      exit 1
    fi
    sleep 5
  done
  echo "[serve] head topology has $expected stage(s)"
}

remote_setup_cmd() {
  local remote_dir
  local uv_bin
  remote_dir="$(remote_quote "$REMOTE_MOLINK_DIR")"
  uv_bin="$(remote_quote "$UV_BIN")"

  cat <<EOF
set -e
cd $remote_dir
echo "[node] \$(hostname) cwd=\$(pwd)"
$uv_bin sync
test -x .venv/bin/python
.venv/bin/python -c 'import molinkv1'
EOF
}

layer_start_for_stage() {
  local stage="$1"
  echo $((stage * MODEL_LAYERS / NUM_STAGES))
}

layer_end_for_stage() {
  local stage="$1"
  if [[ "$stage" -eq $((NUM_STAGES - 1)) ]]; then
    echo "-1"
  else
    echo $(((stage + 1) * MODEL_LAYERS / NUM_STAGES))
  fi
}

api_port_for_stage() {
  local stage="$1"
  if [[ "$stage" -eq 0 ]]; then
    echo "$API_PORT"
  else
    echo $((WORKER_API_PORT_BASE + stage - 1))
  fi
}

grpc_port_for_stage() {
  local stage="$1"
  echo $((GRPC_PORT_BASE + stage))
}

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

SERVER_PIDS=()
SERVER_PID_LABELS=()
SERVER_PID_LOGS=()

cleanup() {
  local exit_code=$?
  local host
  for pid in "${SERVER_PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  for host in "${SETUP_HOSTS[@]}"; do
    if [[ "$host" == "$HEAD_HOST" ]]; then
      pkill -f "molinkv1.entrypoints.api_server" 2>/dev/null || true
    else
      run_remote "$host" "pkill -f molinkv1.entrypoints.api_server || true" >/dev/null 2>&1 || true
    fi
  done
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

if [[ "$CLEANUP_EXISTING_SERVERS" != "0" ]]; then
  echo "[cleanup] stopping existing MoLink API servers before launch"
  for host in "${SETUP_HOSTS[@]}"; do
    if [[ "$host" == "$HEAD_HOST" ]]; then
      pkill -f "molinkv1.entrypoints.api_server" 2>/dev/null || true
    else
      run_remote "$host" "pkill -f molinkv1.entrypoints.api_server || true" >/dev/null 2>&1 || true
    fi
  done
  sleep 2
fi

echo "[setup] working directory: $SCRIPT_DIR"
echo "[setup] log directory: $RUN_LOG_DIR"
echo "[setup] hosts: ${SETUP_HOSTS[*]}"
echo "[setup] stage placement: ${STAGE_LABELS[*]}"
echo "[setup] stages: $NUM_STAGES"
echo "[setup] model layers: $MODEL_LAYERS"

for host in "${SETUP_HOSTS[@]}"; do
  echo "[setup] preparing node: $host"
  if [[ "$host" == "$HEAD_HOST" ]]; then
    bash -lc "$(remote_setup_cmd)"
  else
    run_remote "$host" "$(remote_setup_cmd)"
  fi
done

source "$SCRIPT_DIR/.venv/bin/activate"

{
  echo "# framework: MoLink"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# model: $MODEL"
  echo "# model_layers: $MODEL_LAYERS"
  echo "# hosts: ${SETUP_HOSTS[*]}"
  echo "# stage_placement: ${STAGE_LABELS[*]}"
  echo "# stages: $NUM_STAGES"
  echo "# head_host: $HEAD_HOST"
  echo "# head_address: $HEAD_ADDRESS"
  echo "# api_port: $API_PORT"
  echo "# grpc_port_base: $GRPC_PORT_BASE"
  echo "# max_model_len: $MAX_MODEL_LEN"
  echo "# gpu_memory_utilization: $GPU_MEMORY_UTILIZATION"
  echo "# enforce_eager: $MOLINK_ENFORCE_EAGER"
  echo "# stage_start_timeout_sec: $STAGE_START_TIMEOUT_SEC"
  echo "# pipeline_ready_grace_sec: $PIPELINE_READY_GRACE_SEC"
  echo "# random_input_len: $RANDOM_INPUT_LEN"
  echo "# random_output_len: $RANDOM_OUTPUT_LEN"
  echo "# num_prompts: $NUM_PROMPTS"
  echo "# num_prompts_per_concurrency: $NUM_PROMPTS_PER_CONCURRENCY"
  echo "# max_concurrencies: ${CONCURRENCY_LIST[*]}"
  echo "# ignore_eos: $IGNORE_EOS"
  printf "started_at\tended_at\tduration_sec\tmodel\tmax_concurrency\tstatus\texit_code\tserver_logs\tbench_log\tresult_json\n"
} > "$SUMMARY_LOG"

# shellcheck disable=SC2206
SERVER_EXTRA=($MOLINK_EXTRA_SERVER_ARGS)
# shellcheck disable=SC2206
BENCH_EXTRA=($VLLM_EXTRA_BENCH_ARGS)

HEAD_GRPC_PORT="$(grpc_port_for_stage 0)"
HEAD_PEER="$HEAD_ADDRESS:$HEAD_GRPC_PORT"
SERVER_LOGS=()
STAGE_API_PORTS=()
STAGE_GRPC_PORTS=()

for idx in "${!STAGE_HOSTS[@]}"; do
  host="${STAGE_HOSTS[$idx]}"
  gpu_id="${STAGE_GPU_IDS[$idx]}"
  stage_label="${STAGE_LABELS[$idx]}"
  start_layer="$(layer_start_for_stage "$idx")"
  end_layer="$(layer_end_for_stage "$idx")"
  api_port="$(api_port_for_stage "$idx")"
  grpc_port="$(grpc_port_for_stage "$idx")"
  safe_stage_label="${stage_label//[^A-Za-z0-9_.-]/_}"
  server_log="$RUN_LOG_DIR/server_stage_${idx}_${safe_stage_label}.stdout.log"
  SERVER_LOGS+=("$server_log")
  STAGE_API_PORTS+=("$api_port")
  STAGE_GRPC_PORTS+=("$grpc_port")

  server_cmd=(
    "$PYTHON_BIN" -m molinkv1.entrypoints.api_server
    --model "$MODEL"
    --molink-enabled
    --molink-grpc-port "$grpc_port"
    --molink-start-layer "$start_layer"
    --molink-end-layer "$end_layer"
    --port "$api_port"
    --max-model-len "$MAX_MODEL_LEN"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    "${SERVER_EXTRA[@]}"
  )

  if [[ "$MOLINK_ENFORCE_EAGER" != "0" ]]; then
    server_cmd+=(--enforce-eager)
  fi

  if [[ "$idx" -gt 0 ]]; then
    server_cmd+=(--molink-initial-peer "$HEAD_PEER")
  fi

  echo "[serve] starting stage=$idx host=$host gpu=$gpu_id layers=${start_layer}:${end_layer} api_port=$api_port grpc_port=$grpc_port"
  {
    printf "# host: %s\n" "$host"
    printf "# cuda_visible_devices: %s\n" "$gpu_id"
    printf "# layers: %s:%s\n" "$start_layer" "$end_layer"
    printf "# command:"
    printf " CUDA_VISIBLE_DEVICES=%q" "$gpu_id"
    printf " %q" "${server_cmd[@]}"
    echo
    echo
  } > "$server_log"

  if [[ "$host" == "$HEAD_HOST" ]]; then
    (
      cd "$REMOTE_MOLINK_DIR"
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      "${server_cmd[@]}"
    ) >> "$server_log" 2>&1 &
  else
    remote_dir="$(remote_quote "$REMOTE_MOLINK_DIR")"
    remote_command="cd $remote_dir && CUDA_VISIBLE_DEVICES=$(remote_quote "$gpu_id") $(printf "%q " "${server_cmd[@]}")"
    ssh "${SSH_ARGS[@]}" "$host" "bash -lc $(remote_quote "$remote_command")" >> "$server_log" 2>&1 &
  fi
  SERVER_PIDS+=("$!")
  SERVER_PID_LABELS+=("stage=$idx host=$host gpu=$gpu_id grpc_port=$grpc_port")
  SERVER_PID_LOGS+=("$server_log")

  if [[ "$idx" -eq 0 ]]; then
    wait_for_stage_grpc "$idx" "$host" "$grpc_port" "$!" "$server_log"
    wait_for_topology_count 1 "$idx" "$!" "$server_log"
    if [[ "$NUM_STAGES" -gt 1 && "$SERVER_START_STAGGER_SEC" -gt 0 ]]; then
      sleep "$SERVER_START_STAGGER_SEC"
    fi
  fi
done

echo "[serve] waiting for all MoLink stages"
deadline=$((SECONDS + SERVER_START_TIMEOUT_SEC))
while true; do
  ready=1
  grpc_ready=1
  not_ready_stages=()

  for idx in "${!STAGE_HOSTS[@]}"; do
    host="${STAGE_HOSTS[$idx]}"
    grpc_port="${STAGE_GRPC_PORTS[$idx]}"
    if [[ "$host" == "$HEAD_HOST" ]]; then
      if ! check_local_tcp "127.0.0.1" "$grpc_port" && ! check_local_tcp "$HEAD_ADDRESS" "$grpc_port"; then
        ready=0
        grpc_ready=0
        not_ready_stages+=("stage=$idx host=$host grpc_port=$grpc_port")
      fi
    else
      if ! check_remote_tcp "$host" "$grpc_port"; then
        ready=0
        grpc_ready=0
        not_ready_stages+=("stage=$idx host=$host grpc_port=$grpc_port")
      fi
    fi
  done

  if [[ "$grpc_ready" -eq 1 ]] && ! check_head_topology_count "$NUM_STAGES"; then
    ready=0
    not_ready_stages+=("head_topology_count<$NUM_STAGES")
  fi

  if [[ "$grpc_ready" -eq 1 ]] && check_head_topology_count "$NUM_STAGES"; then
    if ! check_local_http_models "127.0.0.1" "$API_PORT"; then
      ready=0
    fi
  fi

  if [[ "$ready" -eq 1 ]]; then
    break
  fi

  for pid_idx in "${!SERVER_PIDS[@]}"; do
    pid="${SERVER_PIDS[$pid_idx]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[serve] server process exited before readiness: ${SERVER_PID_LABELS[$pid_idx]}" >&2
      echo "[serve] log: ${SERVER_PID_LOGS[$pid_idx]}" >&2
      tail -n 40 "${SERVER_PID_LOGS[$pid_idx]}" >&2 || true
      exit 1
    fi
  done
  if [[ "$SECONDS" -ge "$deadline" ]]; then
    echo "[serve] timed out waiting for all MoLink stages. Not ready: ${not_ready_stages[*]:-head API}" >&2
    echo "[serve] see $RUN_LOG_DIR/server_stage_*.stdout.log" >&2
    exit 1
  fi
  sleep 5
done

if [[ "$PIPELINE_READY_GRACE_SEC" -gt 0 ]]; then
  echo "[serve] all stages are reachable; waiting ${PIPELINE_READY_GRACE_SEC}s for topology to settle"
  sleep "$PIPELINE_READY_GRACE_SEC"
fi

failed=0
server_logs_joined="$(IFS=,; echo "${SERVER_LOGS[*]}")"

for max_concurrency in "${CONCURRENCY_LIST[@]}"; do
  if [[ "$NUM_PROMPTS_PER_CONCURRENCY" != "0" ]]; then
    run_num_prompts="$max_concurrency"
  else
    run_num_prompts="$NUM_PROMPTS"
  fi

  bench_log="$RUN_LOG_DIR/bench_concurrency_${max_concurrency}.stdout.log"
  result_json="$RUN_LOG_DIR/bench_concurrency_${max_concurrency}.json"

  bench_cmd=(
    "$VLLM_BIN" bench serve
    --backend openai
    --model "$SERVED_MODEL_NAME"
    --base-url "http://127.0.0.1:$API_PORT"
    --endpoint /v1/completions
    --dataset-name random
    --num-prompts "$run_num_prompts"
    --random-input-len "$RANDOM_INPUT_LEN"
    --random-output-len "$RANDOM_OUTPUT_LEN"
    --random-range-ratio 0
    --request-rate "$REQUEST_RATE"
    --max-concurrency "$max_concurrency"
    --save-result
    --result-dir "$RUN_LOG_DIR"
    --result-filename "$(basename "$result_json")"
  )

  if [[ "$IGNORE_EOS" != "0" ]]; then
    bench_cmd+=(--ignore-eos)
  fi

  bench_cmd+=("${BENCH_EXTRA[@]}")

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"
  echo "[bench] running benchmark max_concurrency=$max_concurrency num_prompts=$run_num_prompts"
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
    "$server_logs_joined" "$bench_log" "$result_json" >> "$SUMMARY_LOG"
done

echo "[done] summary: $SUMMARY_LOG"
if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

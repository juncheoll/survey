#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRAMEWORKS="${FRAMEWORKS:-vLLM FlexGen MoLink}"
GPU_COUNTS="${GPU_COUNTS:-8}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"
SINGLE_NODE_HOST="${SINGLE_NODE_HOST:-127.0.0.1}"

COMMON_HEAD_ADDRESS="${HEAD_ADDRESS:-${HEAD_IP:-$SINGLE_NODE_HOST}}"
COMMON_HEAD_IP="${HEAD_IP:-${HEAD_ADDRESS:-$SINGLE_NODE_HOST}}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/single_node_multi_gpu"
  else
    LOG_DIR="$SCRIPT_DIR/logs/single_node_multi_gpu"
  fi
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

{
  echo "# benchmark_scope: single_node_multi_gpu"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# frameworks: $FRAMEWORKS"
  echo "# gpu_counts: $GPU_COUNTS"
  echo "# single_node_host: $SINGLE_NODE_HOST"
  echo "# head_address: $COMMON_HEAD_ADDRESS"
  echo "# head_ip: $COMMON_HEAD_IP"
  printf "started_at\tended_at\tduration_sec\tframework\tgpu_count\tstatus\texit_code\tstdout_log\tframework_log_dir\n"
} > "$SUMMARY_LOG"

run_framework_gpu_count() {
  local framework="$1"
  local gpu_count="$2"
  local script_path
  local framework_log_dir
  local stdout_log
  local started_at
  local started_sec
  local ended_at
  local ended_sec
  local duration_sec
  local exit_code
  local status

  case "$framework" in
    vLLM|VLLM|vllm)
      framework="vLLM"
      script_path="$SCRIPT_DIR/vLLM/run_vllm_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/vLLM/gpu${gpu_count}"
      ;;
    FlexGen|flexgen)
      framework="FlexGen"
      script_path="$SCRIPT_DIR/FlexGen/run_flexgen_dist_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/FlexGen/gpu${gpu_count}"
      ;;
    MoLink|molink)
      framework="MoLink"
      script_path="$SCRIPT_DIR/MoLink/run_molink_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/MoLink/gpu${gpu_count}"
      ;;
    *)
      echo "[skip] unknown framework: $framework" >&2
      return 2
      ;;
  esac

  stdout_log="$RUN_LOG_DIR/${framework}_gpu${gpu_count}.stdout.log"
  mkdir -p "$framework_log_dir"

  if [[ ! "$gpu_count" =~ ^[0-9]+$ ]] || [[ "$gpu_count" -lt 1 ]]; then
    echo "[run] invalid gpu count for $framework: $gpu_count"
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$framework" "$gpu_count" "invalid_gpu_count" "2" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"
    return 2
  fi

  if [[ ! -x "$script_path" ]]; then
    echo "[run] missing executable script for $framework: $script_path"
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$framework" "$gpu_count" "missing_script" "127" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"
    return 127
  fi

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] $framework gpu_count=$gpu_count"
  {
    echo "# started_at: $started_at"
    echo "# framework: $framework"
    echo "# gpu_count: $gpu_count"
    echo "# script: $script_path"
    echo "# framework_log_dir: $framework_log_dir"
    echo

    case "$framework" in
      vLLM)
        env \
          HOSTFILE="" \
          HOSTS="" \
          GPUS_PER_NODE="$gpu_count" \
          TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-$gpu_count}" \
          PIPELINE_PARALLEL_SIZE="${PIPELINE_PARALLEL_SIZE:-1}" \
          HEAD_ADDRESS="$COMMON_HEAD_ADDRESS" \
          LOG_DIR="$framework_log_dir" \
          "$script_path"
        ;;
      FlexGen)
        env \
          HOSTFILE="" \
          HOSTS="${FLEXGEN_HOSTS:-}" \
          GPUS_PER_NODE="$gpu_count" \
          HEAD_IP="$COMMON_HEAD_IP" \
          LOG_DIR="$framework_log_dir" \
          "$script_path"
        ;;
      MoLink)
        env \
          HOSTFILE="" \
          HOSTS="${MOLINK_HOSTS:-$SINGLE_NODE_HOST}" \
          GPUS_PER_NODE="$gpu_count" \
          HEAD_ADDRESS="$COMMON_HEAD_ADDRESS" \
          LOG_DIR="$framework_log_dir" \
          "$script_path"
        ;;
    esac
  } > "$stdout_log" 2>&1
  exit_code=$?

  ended_at="$(date -Iseconds)"
  ended_sec="$(date +%s)"
  duration_sec=$((ended_sec - started_sec))

  if [[ "$exit_code" -eq 0 ]]; then
    status="success"
    echo "[run] success: $framework gpu_count=$gpu_count (${duration_sec}s)"
  else
    status="failed"
    echo "[run] failed: $framework gpu_count=$gpu_count (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$framework" "$gpu_count" "$status" "$exit_code" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"

  return "$exit_code"
}

failed=0

# shellcheck disable=SC2206
FRAMEWORK_LIST=($FRAMEWORKS)
# shellcheck disable=SC2206
GPU_COUNT_LIST=($GPU_COUNTS)

for gpu_count in "${GPU_COUNT_LIST[@]}"; do
  for framework in "${FRAMEWORK_LIST[@]}"; do
    if ! run_framework_gpu_count "$framework" "$gpu_count"; then
      failed=$((failed + 1))
      if [[ "$STOP_ON_FAILURE" != "0" ]]; then
        break 2
      fi
    fi
  done
done

{
  echo
  echo "# completed_at: $(date -Iseconds)"
  echo "# failed_runs: $failed"
} >> "$SUMMARY_LOG"

echo "[done] failed_runs=$failed"
echo "[done] summary: $SUMMARY_LOG"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRAMEWORKS="${FRAMEWORKS:-vLLM FlexGen MoLink}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/multi_node"
  else
    LOG_DIR="$SCRIPT_DIR/logs/multi_node"
  fi
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

COMMON_HEAD_ADDRESS="${HEAD_ADDRESS:-${HEAD_IP:-}}"
COMMON_HEAD_IP="${HEAD_IP:-${HEAD_ADDRESS:-}}"

{
  echo "# benchmark_scope: multi_node"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# frameworks: $FRAMEWORKS"
  echo "# hostfile: ${HOSTFILE:-}"
  echo "# hosts: ${HOSTS:-}"
  echo "# head_address: ${COMMON_HEAD_ADDRESS:-}"
  echo "# head_ip: ${COMMON_HEAD_IP:-}"
  printf "started_at\tended_at\tduration_sec\tframework\tstatus\texit_code\tstdout_log\tframework_log_dir\thost_source\n"
} > "$SUMMARY_LOG"

resolve_host_source() {
  local hostfile="$1"
  local hosts="$2"

  if [[ -n "$hostfile" ]]; then
    echo "hostfile:$hostfile"
  elif [[ -n "$hosts" ]]; then
    echo "hosts:$hosts"
  else
    echo ""
  fi
}

absolute_path() {
  local path="$1"

  if [[ -z "$path" || "$path" == /* ]]; then
    echo "$path"
  else
    echo "$SCRIPT_DIR/$path"
  fi
}

run_framework() {
  local framework="$1"
  local script_path
  local framework_log_dir
  local fw_hostfile
  local fw_hosts
  local host_source
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
      framework_log_dir="$RUN_LOG_DIR/vLLM"
      fw_hostfile="${VLLM_HOSTFILE:-${HOSTFILE:-}}"
      fw_hosts="${VLLM_HOSTS:-${HOSTS:-}}"
      ;;
    FlexGen|flexgen)
      framework="FlexGen"
      script_path="$SCRIPT_DIR/FlexGen/run_flexgen_dist_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/FlexGen"
      fw_hostfile="${FLEXGEN_HOSTFILE:-${HOSTFILE:-}}"
      fw_hosts="${FLEXGEN_HOSTS:-${HOSTS:-}}"
      ;;
    MoLink|molink)
      framework="MoLink"
      script_path="$SCRIPT_DIR/MoLink/run_molink_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/MoLink"
      fw_hostfile="${MOLINK_HOSTFILE:-${HOSTFILE:-}}"
      fw_hosts="${MOLINK_HOSTS:-${HOSTS:-}}"
      ;;
    *)
      echo "[skip] unknown framework: $framework" >&2
      return 2
      ;;
  esac

  fw_hostfile="$(absolute_path "$fw_hostfile")"
  host_source="$(resolve_host_source "$fw_hostfile" "$fw_hosts")"
  stdout_log="$RUN_LOG_DIR/${framework}.stdout.log"
  mkdir -p "$framework_log_dir"

  if [[ -z "$host_source" ]]; then
    echo "[run] missing host info for $framework. Set HOSTFILE/HOSTS or ${framework^^}_HOSTFILE/${framework^^}_HOSTS."
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$framework" "missing_hosts" "2" "$stdout_log" "$framework_log_dir" "" >> "$SUMMARY_LOG"
    return 2
  fi

  if [[ -n "$fw_hostfile" && ! -f "$fw_hostfile" ]]; then
    echo "[run] hostfile does not exist for $framework: $fw_hostfile"
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$framework" "missing_hostfile" "2" "$stdout_log" "$framework_log_dir" "$host_source" >> "$SUMMARY_LOG"
    return 2
  fi

  if [[ ! -x "$script_path" ]]; then
    echo "[run] missing executable script for $framework: $script_path"
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$framework" "missing_script" "127" "$stdout_log" "$framework_log_dir" "$host_source" >> "$SUMMARY_LOG"
    return 127
  fi

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] $framework ($host_source)"
  {
    echo "# started_at: $started_at"
    echo "# framework: $framework"
    echo "# script: $script_path"
    echo "# framework_log_dir: $framework_log_dir"
    echo "# host_source: $host_source"
    echo
    env \
      HOSTFILE="$fw_hostfile" \
      HOSTS="$fw_hosts" \
      HEAD_ADDRESS="$COMMON_HEAD_ADDRESS" \
      HEAD_IP="$COMMON_HEAD_IP" \
      LOG_DIR="$framework_log_dir" \
      "$script_path"
  } > "$stdout_log" 2>&1
  exit_code=$?

  ended_at="$(date -Iseconds)"
  ended_sec="$(date +%s)"
  duration_sec=$((ended_sec - started_sec))

  if [[ "$exit_code" -eq 0 ]]; then
    status="success"
    echo "[run] success: $framework (${duration_sec}s)"
  else
    status="failed"
    echo "[run] failed: $framework (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$framework" "$status" "$exit_code" "$stdout_log" "$framework_log_dir" "$host_source" >> "$SUMMARY_LOG"

  return "$exit_code"
}

failed=0

# shellcheck disable=SC2206
FRAMEWORK_LIST=($FRAMEWORKS)

for framework in "${FRAMEWORK_LIST[@]}"; do
  if ! run_framework "$framework"; then
    failed=$((failed + 1))
    if [[ "$STOP_ON_FAILURE" != "0" ]]; then
      break
    fi
  fi
done

{
  echo
  echo "# completed_at: $(date -Iseconds)"
  echo "# failed_frameworks: $failed"
} >> "$SUMMARY_LOG"

echo "[done] failed_frameworks=$failed"
echo "[done] summary: $SUMMARY_LOG"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

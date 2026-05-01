#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRAMEWORKS="${FRAMEWORKS:-FlexGen ZeRO-Inference SubSpec}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/single_node"
  else
    LOG_DIR="$SCRIPT_DIR/logs/single_node"
  fi
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

{
  echo "# benchmark_scope: single_node"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# frameworks: $FRAMEWORKS"
  printf "started_at\tended_at\tduration_sec\tframework\tstatus\texit_code\tstdout_log\tframework_log_dir\n"
} > "$SUMMARY_LOG"

run_framework() {
  local framework="$1"
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
    FlexGen)
      script_path="$SCRIPT_DIR/FlexGen/run_flexgen_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/FlexGen"
      ;;
    ZeRO-Inference|ZeroInference|ZeRO)
      framework="ZeRO-Inference"
      script_path="$SCRIPT_DIR/ZeRO-Inference/run_inference_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/ZeRO-Inference"
      ;;
    SubSpec)
      script_path="$SCRIPT_DIR/SubSpec/run_subspec_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/SubSpec"
      ;;
    PowerInfer|powerinfer)
      framework="PowerInfer"
      script_path="$SCRIPT_DIR/PowerInfer/run_powerinfer_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/PowerInfer"
      ;;
    *)
      echo "[skip] unknown framework: $framework" >&2
      return 2
      ;;
  esac

  stdout_log="$RUN_LOG_DIR/${framework}.stdout.log"
  mkdir -p "$framework_log_dir"

  if [[ ! -x "$script_path" ]]; then
    echo "[run] missing executable script for $framework: $script_path"
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$framework" "missing_script" "127" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"
    return 127
  fi

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] $framework"
  {
    echo "# started_at: $started_at"
    echo "# framework: $framework"
    echo "# script: $script_path"
    echo "# framework_log_dir: $framework_log_dir"
    echo
    LOG_DIR="$framework_log_dir" "$script_path"
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

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$framework" "$status" "$exit_code" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"

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

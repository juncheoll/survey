#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FRAMEWORKS="${FRAMEWORKS:-SpecExec SubSpec PowerInfer}"
STOP_ON_FAILURE="${STOP_ON_FAILURE:-0}"

TEST_INPUT_TOKENS="${TEST_INPUT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
IGNORE_EOS="${IGNORE_EOS:-1}"

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/temp_spec_benchmarks"
  else
    LOG_DIR="$SCRIPT_DIR/logs/temp_spec_benchmarks"
  fi
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

{
  echo "# benchmark_scope: temp_spec_benchmarks"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# frameworks: $FRAMEWORKS"
  echo "# test_input_tokens: $TEST_INPUT_TOKENS"
  echo "# max_new_tokens: $MAX_NEW_TOKENS"
  echo "# ignore_eos: $IGNORE_EOS"
  printf "started_at\tended_at\tduration_sec\tframework\tstatus\texit_code\tstdout_log\tframework_log_dir\n"
} > "$SUMMARY_LOG"

run_specexec() {
  local framework="SpecExec"
  local spec_dir="$SCRIPT_DIR/SpecExec"
  local work_dir="$spec_dir/specexec"
  local framework_log_dir="$RUN_LOG_DIR/$framework"
  local stdout_log="$RUN_LOG_DIR/${framework}.stdout.log"
  local started_at started_sec ended_at ended_sec duration_sec exit_code status
  local max_budget="${SPECEXEC_MAX_BUDGET:-128}"
  local temperature="${SPECEXEC_TEMPERATURE:-0.6}"
  local top_p="${SPECEXEC_TOP_P:-0.9}"
  local model_0="${SPECEXEC_MODEL_0:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
  local model_1="${SPECEXEC_MODEL_1:-meta-llama/Llama-2-7b-chat-hf}"
  local exp_name="${SPECEXEC_EXP_NAME:-SX_temp_${TEST_INPUT_TOKENS}_${MAX_NEW_TOKENS}}"
  local -a cmd

  mkdir -p "$framework_log_dir"

  if [[ ! -d "$work_dir" ]]; then
    echo "[run] missing SpecExec directory: $work_dir"
    return 127
  fi

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] $framework"
  {
    echo "# started_at: $started_at"
    echo "# framework: $framework"
    echo "# working_dir: $work_dir"
    echo "# framework_log_dir: $framework_log_dir"
    echo
    cd "$spec_dir"
    echo "[setup] running uv sync"
    if ! uv sync; then
      echo "[setup] uv sync failed" >&2
      exit 1
    fi
    cd "$work_dir"

    cmd=(
      "../.venv/bin/python" run_exp.py
      --model_0 "$model_0"
      --model_1 "$model_1"
      --gen_type SpecExecBase
      --temperature "$temperature"
      --top_p "$top_p"
      --max_budget "$max_budget"
      --test-input-tokens "$TEST_INPUT_TOKENS"
      --max-new-tokens "$MAX_NEW_TOKENS"
      --n_tests 1
      --offload
      --save_dir "$framework_log_dir"
      --exp_name "$exp_name"
    )
    if [[ "$IGNORE_EOS" != "0" ]]; then
      cmd+=(--ignore-eos)
    fi

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
    echo "[run] success: $framework (${duration_sec}s)"
  else
    status="failed"
    echo "[run] failed: $framework (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$framework" "$status" "$exit_code" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"

  return "$exit_code"
}

run_script_framework() {
  local framework="$1"
  local script_path framework_log_dir stdout_log
  local started_at started_sec ended_at ended_sec duration_sec exit_code status
  local -a env_args

  case "$framework" in
    SubSpec)
      script_path="$SCRIPT_DIR/SubSpec/run_subspec_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/SubSpec"
      env_args=(
        "LOG_DIR=$framework_log_dir"
        "TEST_INPUT_TOKENS=$TEST_INPUT_TOKENS"
        "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
        "IGNORE_EOS=$IGNORE_EOS"
        "DOWNLOAD_MODELS=${DOWNLOAD_MODELS:-1}"
      )
      ;;
    PowerInfer)
      script_path="$SCRIPT_DIR/PowerInfer/run_powerinfer_benchmark.sh"
      framework_log_dir="$RUN_LOG_DIR/PowerInfer"
      env_args=(
        "LOG_DIR=$framework_log_dir"
        "DOWNLOAD_MODELS=1"
        "PP_LIST=${POWERINFER_PP_LIST:-$TEST_INPUT_TOKENS}"
        "TG_LIST=${POWERINFER_TG_LIST:-$MAX_NEW_TOKENS}"
        "PL_LIST=${POWERINFER_PL_LIST:-1,2,4,8}"
        "MODEL_CACHE_DIR=${POWERINFER_MODEL_CACHE_DIR:-$SCRIPT_DIR/PowerInfer/models}"
      )
      ;;
    *)
      echo "[skip] unknown script framework: $framework" >&2
      return 2
      ;;
  esac

  stdout_log="$RUN_LOG_DIR/${framework}.stdout.log"
  mkdir -p "$framework_log_dir"

  if [[ ! -x "$script_path" ]]; then
    echo "[run] missing executable script for $framework: $script_path"
    started_at="$(date -Iseconds)"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$started_at" "0" "$framework" "missing_script" "127" "$stdout_log" "$framework_log_dir" >> "$SUMMARY_LOG"
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
    printf "# env:"
    printf " %q" "${env_args[@]}"
    echo
    echo
    env "${env_args[@]}" "$script_path"
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

run_framework() {
  local framework="$1"

  case "$framework" in
    SpecExec)
      run_specexec
      ;;
    SubSpec|PowerInfer)
      run_script_framework "$framework"
      ;;
    *)
      echo "[skip] unknown framework: $framework" >&2
      return 2
      ;;
  esac
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

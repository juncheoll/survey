#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBSPEC_DIR="$SCRIPT_DIR/subspec"
CONFIG_DIR="$SUBSPEC_DIR/configs/exp_offloading"

cd "$SCRIPT_DIR"

DEFAULT_MODEL_SIZES=(7b 13b 30b)
DEFAULT_VRAM_LIMITS=(10 12 16)

# Positional args select VRAM limits, e.g. `./run_subspec_benchmark.sh 10 16`.
if [[ "$#" -gt 0 ]]; then
  VRAM_LIMITS=("$@")
else
  # shellcheck disable=SC2206
  VRAM_LIMITS=(${VRAM_LIMITS:-${DEFAULT_VRAM_LIMITS[*]}})
fi

# shellcheck disable=SC2206
MODEL_SIZES=(${MODEL_SIZES:-${DEFAULT_MODEL_SIZES[*]}})

TEST_INPUT_TOKENS="${TEST_INPUT_TOKENS:-1024}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
IGNORE_EOS="${IGNORE_EOS:-1}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SPECEXEC_PROMPTS_FILE="${SPECEXEC_PROMPTS_FILE:-$SCRIPT_DIR/../SpecExec/specexec/data/oasst_prompts.json}"
TEST_INPUT_TEXT="${TEST_INPUT_TEXT:-}"

if [[ -z "${LOG_DIR:-}" ]]; then
  LOG_DIR="$SCRIPT_DIR/logs/subspec_benchmark"
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR"

echo "[setup] working directory: $SCRIPT_DIR"
echo "[setup] subspec directory: $SUBSPEC_DIR"
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

if [[ ! -d "$SUBSPEC_DIR" ]]; then
  echo "[setup] subspec directory not found: $SUBSPEC_DIR" >&2
  exit 1
fi

load_default_test_input_text() {
  local prompts_file="$1"
  [[ -f "$prompts_file" ]] || return 1
  "$PYTHON_BIN" - "$prompts_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    data = json.load(f)

first = data[0]
if isinstance(first, (list, tuple)) and len(first) >= 2:
    print(first[1], end="")
else:
    print(first, end="")
PY
}

if [[ -z "$TEST_INPUT_TEXT" ]]; then
  if TEST_INPUT_TEXT="$(load_default_test_input_text "$SPECEXEC_PROMPTS_FILE")"; then
    echo "[setup] test input text: SpecExec OASST prompt[0] from $SPECEXEC_PROMPTS_FILE"
  else
    echo "[setup] SpecExec prompt file not found; falling back to config test_prompt"
    TEST_INPUT_TEXT=""
  fi
fi

expand_path() {
  local path="$1"
  if [[ "$path" == "~/"* ]]; then
    printf "%s/%s\n" "$HOME" "${path#~/}"
  else
    printf "%s\n" "$path"
  fi
}

config_value() {
  local key="$1"
  local config_file="$2"
  awk -F: -v key="$key" '
    $1 == key {
      sub(/^[[:space:]]+/, "", $2)
      sub(/[[:space:]]+$/, "", $2)
      gsub(/^["'\''"]|["'\''"]$/, "", $2)
      print $2
      exit
    }
  ' "$config_file"
}

model_cache_has_weights() {
  local model_id="$1"
  local cache_dir="$2"
  local repo_dir="$cache_dir/models--${model_id//\//--}"

  [[ -d "$repo_dir/snapshots" ]] || return 1
  find -L "$repo_dir/snapshots" -type f \
    \( -name "*.safetensors" -o -name "pytorch_model*.bin" -o -name "model*.bin" \) \
    -print -quit | grep -q .
}

download_model() {
  local model_id="$1"
  local cache_dir="$2"

  if command -v hf >/dev/null 2>&1; then
    hf download "$model_id" --cache-dir "$cache_dir"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download "$model_id" --cache-dir "$cache_dir"
  else
    echo "[preflight] neither hf nor huggingface-cli is available" >&2
    return 1
  fi
}

ensure_model_cached() {
  local config_file="$1"
  local model_id
  local cache_dir

  model_id="$(config_value "llm_path" "$config_file")"
  cache_dir="$(config_value "model_cache_dir" "$config_file")"
  cache_dir="$(expand_path "${cache_dir:-$HOME/.cache/huggingface/hub}")"

  if [[ -z "$model_id" ]]; then
    echo "[preflight] llm_path is missing in $config_file" >&2
    return 20
  fi

  if model_cache_has_weights "$model_id" "$cache_dir"; then
    echo "[preflight] model weights found in cache: $model_id"
    return 0
  fi

  echo "[preflight] model weights are missing from cache: $model_id"
  echo "[preflight] cache_dir: $cache_dir"

  if [[ "$DOWNLOAD_MODELS" == "0" ]]; then
    echo "[preflight] DOWNLOAD_MODELS=0, so not attempting download" >&2
    return 21
  fi

  echo "[preflight] downloading model with Hugging Face CLI"
  if ! download_model "$model_id" "$cache_dir"; then
    echo "[preflight] download failed: $model_id" >&2
    echo "[preflight] for gated Llama models, run huggingface-cli login or set HF_TOKEN" >&2
    return 22
  fi

  if ! model_cache_has_weights "$model_id" "$cache_dir"; then
    echo "[preflight] download finished but no weight files were found: $model_id" >&2
    return 23
  fi

  echo "[preflight] model weights are ready: $model_id"
}

{
  echo "# framework: SubSpec"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# model_sizes: ${MODEL_SIZES[*]}"
  echo "# vram_limits_gb: ${VRAM_LIMITS[*]}"
  echo "# test_input_tokens: $TEST_INPUT_TOKENS"
  echo "# test_input_text_source: ${SPECEXEC_PROMPTS_FILE}"
  echo "# max_new_tokens: $MAX_NEW_TOKENS"
  echo "# ignore_eos: $IGNORE_EOS"
  echo "# download_models: $DOWNLOAD_MODELS"
  printf "started_at\tended_at\tduration_sec\tmodel_size\tvram_limit_gb\tconfig\tstatus\texit_code\tstdout_log\texperiment_log_dir\n"
} > "$SUMMARY_LOG"

total=0
success=0
failed=0
missing=0

cd "$SUBSPEC_DIR"

run_one() {
  local model_size="$1"
  local vram_limit="${2%gb}"
  vram_limit="${vram_limit%GB}"
  local config_rel="configs/exp_offloading/subspec_sd_llama_${model_size}_vram_${vram_limit}gb.yaml"
  local config_abs="$CONFIG_DIR/subspec_sd_llama_${model_size}_vram_${vram_limit}gb.yaml"
  local stdout_log="$RUN_LOG_DIR/llama_${model_size}_vram_${vram_limit}gb.stdout.log"
  local started_at
  local started_sec
  local ended_at
  local ended_sec
  local duration_sec
  local exit_code
  local status
  local experiment_log_dir
  local -a cmd

  total=$((total + 1))

  if [[ ! -f "$config_abs" ]]; then
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    status="missing_config"
    exit_code=2
    missing=$((missing + 1))
    failed=$((failed + 1))
    {
      echo "Missing config: $config_abs"
    } > "$stdout_log"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$model_size" "$vram_limit" "$config_rel" "$status" "$exit_code" \
      "$stdout_log" "" >> "$SUMMARY_LOG"
    echo "[run] missing config: $config_rel"
    return
  fi

  cmd=(
    "$PYTHON_BIN" -m run.main
    --config "$config_rel"
    --test-input-tokens "$TEST_INPUT_TOKENS"
    --max-new-tokens "$MAX_NEW_TOKENS"
  )

  if [[ -n "$TEST_INPUT_TEXT" ]]; then
    cmd+=(--test-input-text "$TEST_INPUT_TEXT")
  fi

  if [[ "$IGNORE_EOS" != "0" ]]; then
    cmd+=(--ignore-eos)
  fi

  cmd+=(run-test)

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] llama_${model_size} vram=${vram_limit}gb"
  {
    echo "# started_at: $started_at"
    printf "# command:"
    printf " %q" "${cmd[@]}"
    echo
    echo
    ensure_model_cached "$config_abs"
    preflight_exit_code=$?
    if [[ "$preflight_exit_code" -ne 0 ]]; then
      echo "[run] skipped because preflight failed with exit_code=$preflight_exit_code" >&2
      (exit "$preflight_exit_code")
    else
      "${cmd[@]}"
    fi
  } > "$stdout_log" 2>&1
  exit_code=$?

  ended_at="$(date -Iseconds)"
  ended_sec="$(date +%s)"
  duration_sec=$((ended_sec - started_sec))
  experiment_log_dir="$(grep -E '^Log directory:' "$stdout_log" | tail -1 | sed 's/^Log directory:[[:space:]]*//')"

  if [[ "$exit_code" -eq 0 ]]; then
    status="success"
    success=$((success + 1))
    echo "[run] success: llama_${model_size} vram=${vram_limit}gb (${duration_sec}s)"
  else
    status="failed"
    failed=$((failed + 1))
    echo "[run] failed: llama_${model_size} vram=${vram_limit}gb (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$model_size" "$vram_limit" "$config_rel" "$status" "$exit_code" \
    "$stdout_log" "$experiment_log_dir" >> "$SUMMARY_LOG"
}

for model_size in "${MODEL_SIZES[@]}"; do
  for vram_limit in "${VRAM_LIMITS[@]}"; do
    run_one "$model_size" "$vram_limit"
  done
done

{
  echo
  echo "# completed_at: $(date -Iseconds)"
  echo "# total: $total"
  echo "# success: $success"
  echo "# failed: $failed"
  echo "# missing_config: $missing"
} >> "$SUMMARY_LOG"

echo "[done] total=$total success=$success failed=$failed missing_config=$missing"
echo "[done] summary: $SUMMARY_LOG"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi

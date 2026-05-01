#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POWERINFER_DIR="${POWERINFER_DIR:-$SCRIPT_DIR/PowerInfer}"
BUILD_DIR="${BUILD_DIR:-$POWERINFER_DIR/build}"
BENCH_BIN="${BENCH_BIN:-$BUILD_DIR/bin/batched-bench}"
MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$SCRIPT_DIR/models}"

cd "$SCRIPT_DIR"

DEFAULT_MODELS=(
  "PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF"
  "PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF"
  "PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF"
)

MODELS=("${DEFAULT_MODELS[@]}")
if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
fi

RUN_UV_SYNC="${RUN_UV_SYNC:-1}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"
RUN_BUILD="${RUN_BUILD:-1}"
CMAKE_FLAGS="${CMAKE_FLAGS:--DLLAMA_CUBLAS=ON}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
N_KV_MAX="${N_KV_MAX:-2048}"
IS_PP_SHARED="${IS_PP_SHARED:-0}"
VRAM_BUDGET_GB="${VRAM_BUDGET_GB:-9}"
MMQ="${MMQ:-0}"
PP_LIST="${PP_LIST:-}"
TG_LIST="${TG_LIST:-}"
PL_LIST="${PL_LIST:-}"
MODEL_FILE_GLOB="${MODEL_FILE_GLOB:-*.gguf}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
HF_CLI="${HF_CLI:-hf}"

if [[ -z "$PP_LIST" && ( -n "$TG_LIST" || -n "$PL_LIST" ) ]]; then
  echo "PP_LIST must be set when TG_LIST or PL_LIST is set, because batched-bench parses positional workload lists." >&2
  exit 2
fi
if [[ -z "$TG_LIST" && -n "$PL_LIST" ]]; then
  echo "TG_LIST must be set when PL_LIST is set, because batched-bench parses positional workload lists." >&2
  exit 2
fi

if [[ -z "${LOG_DIR:-}" ]]; then
  if [[ -d "/logs" && -w "/logs" ]]; then
    LOG_DIR="/logs/powerinfer"
  else
    LOG_DIR="$SCRIPT_DIR/logs/powerinfer"
  fi
fi

RUN_ID="$(date +"%Y%m%d-%H%M%S")"
RUN_LOG_DIR="$LOG_DIR/$RUN_ID"
SUMMARY_LOG="$RUN_LOG_DIR/summary.tsv"
mkdir -p "$RUN_LOG_DIR" "$MODEL_CACHE_DIR"

echo "[setup] working directory: $SCRIPT_DIR"
echo "[setup] PowerInfer source: $POWERINFER_DIR"
echo "[setup] build directory: $BUILD_DIR"
echo "[setup] log directory: $RUN_LOG_DIR"
echo "[setup] model cache directory: $MODEL_CACHE_DIR"

if [[ "$RUN_UV_SYNC" != "0" ]]; then
  echo "[setup] running uv sync"
  if ! uv sync; then
    echo "[setup] uv sync failed" >&2
    exit 1
  fi
fi

VENV_ACTIVATE="$SCRIPT_DIR/.venv/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[setup] virtualenv activation script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$VENV_ACTIVATE"

if [[ "$INSTALL_REQUIREMENTS" != "0" ]]; then
  echo "[setup] installing PowerInfer requirements"
  if ! python -m pip install -r "$POWERINFER_DIR/requirements.txt"; then
    echo "[setup] pip install failed" >&2
    exit 1
  fi
fi

if [[ "$RUN_BUILD" != "0" || ! -x "$BENCH_BIN" ]]; then
  echo "[setup] configuring PowerInfer"
  # shellcheck disable=SC2206
  CMAKE_FLAG_ARGS=($CMAKE_FLAGS)
  if ! cmake -S "$POWERINFER_DIR" -B "$BUILD_DIR" "${CMAKE_FLAG_ARGS[@]}"; then
    echo "[setup] cmake configure failed" >&2
    exit 1
  fi

  echo "[setup] building PowerInfer"
  if ! cmake --build "$BUILD_DIR" --config Release -j "$BUILD_JOBS"; then
    echo "[setup] cmake build failed" >&2
    exit 1
  fi
fi

if [[ ! -x "$BENCH_BIN" ]]; then
  echo "[setup] batched-bench not found or not executable: $BENCH_BIN" >&2
  exit 1
fi

resolve_model_dir() {
  local model="$1"
  local repo_name="${model##*/}"
  local model_dir="$MODEL_CACHE_DIR/$repo_name"

  if [[ -d "$model" ]]; then
    printf "%s\n" "$model"
    return
  fi

  if [[ -d "$model_dir" ]]; then
    if find "$model_dir" -type f -name "$MODEL_FILE_GLOB" | grep -q .; then
      printf "%s\n" "$model_dir"
      return
    fi
    echo "[download] model directory exists but no GGUF was found; checking $model again" >&2
  fi

  if [[ "$DOWNLOAD_MODELS" == "0" ]]; then
    echo "[download] missing model directory for $model and DOWNLOAD_MODELS=0" >&2
    return 1
  fi

  mkdir -p "$model_dir"
  echo "[download] downloading $model to $model_dir" >&2
  if [[ "$(basename "$HF_CLI")" == "huggingface-cli" ]]; then
    "$HF_CLI" download --resume-download --local-dir "$model_dir" --local-dir-use-symlinks False "$model" >&2
  else
    "$HF_CLI" download "$model" --local-dir "$model_dir" >&2
  fi
  if [[ "$?" -ne 0 ]]; then
    echo "[download] failed: $model" >&2
    return 1
  fi

  printf "%s\n" "$model_dir"
}

resolve_model_file() {
  local model_dir="$1"
  local selected=""

  selected="$(find "$model_dir" -type f -name "$MODEL_FILE_GLOB" | sort | grep -E '\.powerinfer\.gguf$' | head -1 || true)"
  if [[ -z "$selected" ]]; then
    selected="$(find "$model_dir" -type f -name "$MODEL_FILE_GLOB" | sort | head -1 || true)"
  fi

  if [[ -z "$selected" ]]; then
    echo "[model] no GGUF file found under $model_dir with glob $MODEL_FILE_GLOB" >&2
    return 1
  fi

  printf "%s\n" "$selected"
}

{
  echo "# framework: PowerInfer"
  echo "# run_id: $RUN_ID"
  echo "# started_at: $(date -Iseconds)"
  echo "# powerinfer_dir: $POWERINFER_DIR"
  echo "# bench_bin: $BENCH_BIN"
  echo "# models: ${MODELS[*]}"
  echo "# model_cache_dir: $MODEL_CACHE_DIR"
  echo "# n_kv_max: $N_KV_MAX"
  echo "# is_pp_shared: $IS_PP_SHARED"
  echo "# vram_budget_gb: $VRAM_BUDGET_GB"
  echo "# mmq: $MMQ"
  echo "# pp_list: ${PP_LIST:-bench_default}"
  echo "# tg_list: ${TG_LIST:-bench_default}"
  echo "# pl_list: ${PL_LIST:-bench_default}"
  printf "started_at\tended_at\tduration_sec\tmodel\tmodel_path\tstatus\texit_code\tstdout_log\n"
} > "$SUMMARY_LOG"

total=0
success=0
failed=0

run_one() {
  local model="$1"
  local safe_model="${model//\//_}"
  local stdout_log="$RUN_LOG_DIR/${safe_model}.stdout.log"
  local started_at
  local started_sec
  local ended_at
  local ended_sec
  local duration_sec
  local exit_code
  local status
  local model_dir
  local model_path
  local -a cmd

  total=$((total + 1))

  if ! model_dir="$(resolve_model_dir "$model")"; then
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    status="download_failed"
    exit_code=1
    failed=$((failed + 1))
    {
      echo "Failed to resolve or download model: $model"
      echo "See the parent PowerInfer stdout log for the downloader output."
    } > "$stdout_log"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$model" "" "$status" "$exit_code" "$stdout_log" >> "$SUMMARY_LOG"
    echo "[run] download failed: $model"
    return
  fi

  if ! model_path="$(resolve_model_file "$model_dir")"; then
    started_at="$(date -Iseconds)"
    ended_at="$started_at"
    status="missing_model_file"
    exit_code=2
    failed=$((failed + 1))
    {
      echo "No GGUF model file found under: $model_dir"
    } > "$stdout_log"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
      "$started_at" "$ended_at" "0" "$model" "" "$status" "$exit_code" "$stdout_log" >> "$SUMMARY_LOG"
    echo "[run] missing GGUF: $model"
    return
  fi

  cmd=(
    "$BENCH_BIN"
    "$model_path"
    "$N_KV_MAX"
    "$IS_PP_SHARED"
    "$VRAM_BUDGET_GB"
    "$MMQ"
  )

  if [[ -n "$PP_LIST" ]]; then
    cmd+=("$PP_LIST")
  fi
  if [[ -n "$TG_LIST" ]]; then
    cmd+=("$TG_LIST")
  fi
  if [[ -n "$PL_LIST" ]]; then
    cmd+=("$PL_LIST")
  fi

  started_at="$(date -Iseconds)"
  started_sec="$(date +%s)"

  echo "[run] $model"
  {
    echo "# started_at: $started_at"
    echo "# model_dir: $model_dir"
    echo "# model_path: $model_path"
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
    echo "[run] success: $model (${duration_sec}s)"
  else
    status="failed"
    failed=$((failed + 1))
    echo "[run] failed: $model (${duration_sec}s, exit_code=$exit_code)"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$started_at" "$ended_at" "$duration_sec" "$model" "$model_path" "$status" "$exit_code" "$stdout_log" >> "$SUMMARY_LOG"
}

for model in "${MODELS[@]}"; do
  run_one "$model"
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

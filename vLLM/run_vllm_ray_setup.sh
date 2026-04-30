#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_VLLM_DIR="${REMOTE_VLLM_DIR:-$SCRIPT_DIR}"
UV_BIN="${UV_BIN:-uv}"
RAY_PORT="${RAY_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
SSH_PORT="${SSH_PORT:-}"
SSH_EXTRA_ARGS="${SSH_EXTRA_ARGS:-}"
RAY_STOP_FIRST="${RAY_STOP_FIRST:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"

HOST_NAMES=()
HOST_SLOTS=()

add_host() {
  local host="$1"
  local slots="$2"
  HOST_NAMES+=("$host")
  HOST_SLOTS+=("$slots")
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
  IFS=',' read -r -a HOST_NAMES <<< "$HOSTS"
  for _ in "${HOST_NAMES[@]}"; do
    HOST_SLOTS+=("$GPUS_PER_NODE")
  done
else
  add_host "127.0.0.1" "$GPUS_PER_NODE"
fi

if [[ "${#HOST_NAMES[@]}" -eq 0 ]]; then
  echo "No hosts found. Set HOSTFILE or HOSTS." >&2
  exit 2
fi

HEAD_HOST="${HEAD_HOST:-${HOST_NAMES[0]}}"
HEAD_ADDRESS="${HEAD_ADDRESS:-$(hostname -i | awk '{print $1}')}"

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

remote_setup_cmd() {
  local slots="$1"
  local role="$2"
  local remote_dir
  local uv_bin
  remote_dir="$(remote_quote "$REMOTE_VLLM_DIR")"
  uv_bin="$(remote_quote "$UV_BIN")"

  cat <<EOF
set -e
cd $remote_dir
echo "[node] \$(hostname) role=$role cwd=\$(pwd)"
$uv_bin sync
test -x .venv/bin/ray
if [[ "$RAY_STOP_FIRST" != "0" ]]; then
  .venv/bin/ray stop --force || true
fi
if [[ "$role" == "head" ]]; then
  .venv/bin/ray start --head --node-ip-address "$HEAD_ADDRESS" --port "$RAY_PORT" --dashboard-host 0.0.0.0 --dashboard-port "$RAY_DASHBOARD_PORT" --num-gpus "$slots" --disable-usage-stats
else
  .venv/bin/ray start --address "$HEAD_ADDRESS:$RAY_PORT" --num-gpus "$slots" --disable-usage-stats
fi
.venv/bin/ray status --address "$HEAD_ADDRESS:$RAY_PORT" || true
EOF
}

echo "[setup] Ray head: $HEAD_HOST ($HEAD_ADDRESS:$RAY_PORT)"
echo "[setup] Remote vLLM dir: $REMOTE_VLLM_DIR"

for idx in "${!HOST_NAMES[@]}"; do
  host="${HOST_NAMES[$idx]}"
  slots="${HOST_SLOTS[$idx]}"

  if [[ "$idx" -eq 0 ]]; then
    role="head"
    echo "[setup] starting head: $host slots=$slots"
    bash -lc "$(remote_setup_cmd "$slots" "$role")"
  else
    role="worker"
    echo "[setup] starting worker: $host slots=$slots"
    run_remote "$host" "$(remote_setup_cmd "$slots" "$role")"
  fi
done

echo "[setup] Ray cluster status"
"$REMOTE_VLLM_DIR/.venv/bin/ray" status --address "$HEAD_ADDRESS:$RAY_PORT"

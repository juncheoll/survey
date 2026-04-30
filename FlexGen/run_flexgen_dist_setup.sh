#!/usr/bin/env bash

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REMOTE_FLEXGEN_DIR="${REMOTE_FLEXGEN_DIR:-$SCRIPT_DIR}"
UV_BIN="${UV_BIN:-uv}"
HEAD_IP="${HEAD_IP:-$(hostname -i | awk '{print $1}')}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:-}"

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

REMOTE_DIR_QUOTED="$(printf "%q" "$REMOTE_FLEXGEN_DIR")"
UV_BIN_QUOTED="$(printf "%q" "$UV_BIN")"

REMOTE_CMD="
set -e
cd $REMOTE_DIR_QUOTED
echo \"[node] \$(hostname) cwd=\$(pwd)\"
echo \"[node] running uv sync\"
$UV_BIN_QUOTED sync
test -x .venv/bin/python
.venv/bin/python --version
.venv/bin/python -c 'import flexllmgen'
echo \"[node] flexllmgen import ok\"
"

echo "[setup] hosts: $HOSTS_LABEL"
echo "[setup] remote FlexGen dir: $REMOTE_FLEXGEN_DIR"

mpirun \
  "${MPI_HOST_ARGS[@]}" \
  "${MPI_EXTRA[@]}" \
  --map-by ppr:1:node \
  --oversubscribe \
  bash -lc "$REMOTE_CMD"

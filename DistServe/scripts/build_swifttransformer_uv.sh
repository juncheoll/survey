#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_PATH="${PYTHON_PATH:-$ROOT_DIR/.venv/bin/python}"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/SwiftTransformer/build-uv}"

if [[ ! -x "$PYTHON_PATH" ]]; then
  echo "Python not found at $PYTHON_PATH"
  echo "Run 'uv sync' from $ROOT_DIR first, or set PYTHON_PATH."
  exit 1
fi

NCCL_PATHS="$("$PYTHON_PATH" - <<'PY'
from pathlib import Path
import nvidia.nccl as nccl

root = Path(nccl.__file__).parent
include = root / "include"
library = root / "lib" / "libnccl.so.2"
if not (include / "nccl.h").exists() or not library.exists():
    raise SystemExit(f"NCCL wheel files not found under {root}")
print(include)
print(library)
PY
)"
NCCL_INCLUDE_DIR="$(printf '%s\n' "$NCCL_PATHS" | sed -n '1p')"
NCCL_LIBRARY="$(printf '%s\n' "$NCCL_PATHS" | sed -n '2p')"

MPI_ARGS=()
if command -v mpicxx >/dev/null 2>&1; then
  MPI_ARGS+=("-DMPI_CXX_COMPILER=$(command -v mpicxx)")
fi

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"

cmake -S "$ROOT_DIR/SwiftTransformer" -B "$BUILD_DIR" \
  -DPYTHON_PATH="$PYTHON_PATH" \
  -DNCCL_INCLUDE_DIR="$NCCL_INCLUDE_DIR" \
  -DNCCL_INCLUDE_DIRS="$NCCL_INCLUDE_DIR" \
  -DNCCL_LIBRARIES="$NCCL_LIBRARY" \
  "${MPI_ARGS[@]}"

cmake --build "$BUILD_DIR" -j"$(nproc)"

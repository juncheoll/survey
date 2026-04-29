#!/bin/bash

MY_IPADDR=$(hostname -i)
all_hosts=$MY_IPADDR
N_GPUS=4
N_CORES_PER_GPU=4

# Detect virtual environment (uv venv or conda)
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_PYTHON=$VIRTUAL_ENV/bin/python
elif [ -n "$CONDA_PREFIX" ]; then
    VENV_PYTHON=$CONDA_PREFIX/bin/python
elif [ -d ".venv/bin" ]; then
    VENV_PYTHON=$(pwd)/.venv/bin/python
elif [ -d "venv/bin" ]; then
    VENV_PYTHON=$(pwd)/venv/bin/python
else
    echo "Error: No virtual environment detected!"
    echo "Please activate uv venv or conda environment first:"
    echo "  source .venv/bin/activate  (for uv venv)"
    echo "  conda activate flexllmgen  (for conda)"
    exit 1
fi

# Verify Python executable exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: Python executable not found at $VENV_PYTHON"
    exit 1
fi

echo "Using Python: $VENV_PYTHON"
echo "Python version: $($VENV_PYTHON --version)"

PYTHON_EXEC=$VENV_PYTHON
PYTHON_SCRIPT=flexllmgen.dist_flex_opt

# Default model
MODEL=${1:-facebook/opt-1.3b}

pgrep -fl python | awk '!/dist_flex_opt\.py/{print $1}' | xargs sudo kill

set -x

mpirun \
  --mca btl_tcp_if_exclude lo,docker0 \
  --mca oob_tcp_if_exclude lo,docker0 \
  --map-by ppr:$N_GPUS:node:pe=$N_CORES_PER_GPU --oversubscribe -H $all_hosts \
  --bind-to core -x OMP_NUM_THREADS=$N_CORES_PER_GPU \
  $PYTHON_EXEC -m $PYTHON_SCRIPT \
    --head-ip $MY_IPADDR \
    --port 7777 \
    --use-mpi \
    --model $MODEL \
    --gpu-batch-size 16 \
    --percent 100 0 100 0 100 0 \
    --comm-device gpu


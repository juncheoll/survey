#!/bin/bash

N_GPUS=1
N_NODES=4
N_CORES_PER_GPU=16

# Default Llama model
MODEL=${1:-huggyllama/llama-7b}

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

MY_IPADDR=$(hostname -i)
all_public_ips=$(ray get-worker-ips ~/ray_bootstrap_config.yaml)
for s in $all_public_ips; do
    ssh -o StrictHostKeyChecking=no $s hostname -i > /tmp/$s.ip &
done
wait
for s in $all_public_ips; do
    OTHERS_IPADDR+=($(cat /tmp/$s.ip))
done
ALL_IPADDR=($MY_IPADDR ${OTHERS_IPADDR[@]})
all_hosts=$(echo ${ALL_IPADDR[@]:0:$N_NODES} | sed 's/ /,/g')

PYTHON_EXEC=$VENV_PYTHON
PYTHON_SCRIPT=flexllmgen.dist_flex_opt

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
    --gpu-batch-size 8 \
    --num-gpu-batches 2 \
    --percent 100 0 100 0 100 0 \
    --comm-device gpu \
    --async-comm
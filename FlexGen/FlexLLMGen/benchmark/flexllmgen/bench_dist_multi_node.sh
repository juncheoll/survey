#!/bin/bash

# ============================================================================
# FlexGen Multi-Node OPT Benchmark Script
# 
# NOTE: Run this script ONLY on the HEAD NODE
# MPI will automatically distribute work to worker nodes
# ============================================================================

N_GPUS=1
N_NODES=4
N_CORES_PER_GPU=16

# Default model
MODEL=${1:-facebook/opt-1.3b}

# Detect virtual environment (uv venv or conda)
if [ -n "$VIRTUAL_ENV" ]; then
    VENV_PYTHON=$VIRTUAL_ENV/bin/python
    VENV_BIN=$VIRTUAL_ENV/bin
elif [ -n "$CONDA_PREFIX" ]; then
    VENV_PYTHON=$CONDA_PREFIX/bin/python
    VENV_BIN=$CONDA_PREFIX/bin
elif [ -d ".venv/bin" ]; then
    VENV_PYTHON=$(pwd)/.venv/bin/python
    VENV_BIN=$(pwd)/.venv/bin
elif [ -d "venv/bin" ]; then
    VENV_PYTHON=$(pwd)/venv/bin/python
    VENV_BIN=$(pwd)/venv/bin
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

# Check Ray is installed in the virtual environment
if [ ! -f "$VENV_BIN/ray" ]; then
    echo "Error: Ray is not installed in virtual environment at $VENV_BIN"
    echo "Install with: pip install ray"
    exit 1
fi

RAY_EXEC=$VENV_BIN/ray

# Check Ray cluster status
echo "Checking Ray cluster status..."
if ! command -v ray &> /dev/null; then
    echo "Error: Ray is not installed. Install with: pip install ray"
    exit 1
fi

# Get worker IPs from running Ray cluster
echo "Getting worker IPs from Ray cluster..."
MY_IPADDR=$(hostname -i)

# Try to get worker IPs - works if Ray cluster is running locally or via RAY_ADDRESS env var
if [ -f ~/ray_bootstrap_config.yaml ]; then
    # Use bootstrap config if it exists - use $RAY_EXEC from venv
    all_public_ips=$($RAY_EXEC get-worker-ips ~/ray_bootstrap_config.yaml 2>/dev/null)
else
    # Try to get from running cluster (requires RAY_ADDRESS or local cluster)
    # RAY_ADDRESS can be set like: export RAY_ADDRESS=HEAD_IP:6379
    all_public_ips=$($RAY_EXEC get-worker-ips 2>/dev/null)
    
    if [ -z "$all_public_ips" ]; then
        # If still no worker IPs, try with explicit ray address
        if [ -z "$RAY_ADDRESS" ]; then
            echo "Error: Could not get worker IPs from Ray cluster"
            echo ""
            echo "Solutions:"
            echo "1. Start Ray cluster and set RAY_ADDRESS environment variable:"
            echo "   export RAY_ADDRESS=HEAD_IP:6379"
            echo ""
            echo "2. Or create ~/ray_bootstrap_config.yaml with your cluster info"
            echo ""
            echo "3. Or provide Ray address as environment variable before running:"
            echo "   RAY_ADDRESS=HEAD_IP:6379 ./bench_dist_multi_node.sh"
            exit 1
        fi
    fi
fi

if [ -z "$all_public_ips" ]; then
    echo "Error: Could not get worker IPs from Ray cluster at $RAY_ADDRESS"
    echo "Make sure Ray cluster is running: ray status"
    exit 1
fi

echo "Head node IP: $MY_IPADDR"
echo "Worker node IPs: $all_public_ips"
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
    --gpu-batch-size 16 \
    --num-gpu-batches 2 \
    --percent 100 0 100 0 100 0 \
    --comm-device gpu \
    --async-comm


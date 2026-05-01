### docker build
```
docker build -f Dockerfile -t flexgen:cu128 .
```

### docker run
```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v ./logs:/logs \
  --name flexgen \
  flexgen:cu128
```

### create & activate env
```
uv venv /opt/venvs/flexgen
source /opt/venvs/flexgen/bin/activate
```

### install FlexGen
```
git clone https://github.com/FMInference/FlexLLMGen.git
cd FlexLLMGen
uv pip install -e .
```

### run FlexGen

#### OPT Model
```
python3 -m flexllmgen.flex_opt --model facebook/opt-6.7b --percent 0 100 100 0 100 0 --gpu-batch-size 16 --compress-weight
```

#### Llama Model (NEW!)
```
python3 -m flexllmgen.flex_opt --model meta-llama/Llama-2-13b-hf --percent 0 100 100 0 100 0 --path _DUMMY_
```

#### Benchmark Llama Models
```
./run_flexgen_benchmark.sh
```

The script runs `uv sync`, activates `.venv`, and benchmarks:
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`
- `huggyllama/llama-30b`

Each model is tested with GPU batch sizes `1`, `2`, `4`, and `8`, using `--prompt-len 1024` and `--gen-len 256`.
For each batch size, the script runs both the default mode and the weight-compressed mode with `--compress-weight`.
The weight-compressed mode also includes `meta-llama/Llama-2-70b-hf`.

Logs are written to `/logs/flexgen/<run-id>` inside the Docker container, or `./logs/flexgen/<run-id>` outside the container.


#### Multi-GPU(Multi-Node)
```
python -m flexllmgen.dist_flex_opt --head-ip 192.168.79.22 --port 7777 --rank 0 --local-rank 0 --world-size 2 --model meta-llama/Llama-2-13b-hf --path _DUMMY_ --percent 80 20 100 0 100 0 --comm-device gpu --async-comm
```

#### Distributed Benchmark From Head Node
Host information can be provided in either of these ways:

```
HOSTS="192.168.79.22,192.168.79.23" GPUS_PER_NODE=1 ./run_flexgen_dist_benchmark.sh
```

or with an OpenMPI hostfile:

```
cp hosts.example hosts
HOSTFILE=hosts GPUS_PER_NODE=1 ./run_flexgen_dist_benchmark.sh
```

Every node must have this repository at the same path. By default, the script assumes the current FlexGen path exists on every node. Override it if needed:

```
REMOTE_FLEXGEN_DIR=/workspace/FlexGen \
HOSTS="192.168.79.22,192.168.79.23" \
GPUS_PER_NODE=1 \
./run_flexgen_dist_benchmark.sh
```

The distributed benchmark script runs `run_flexgen_dist_setup.sh` first. The setup step uses `mpirun` to run `uv sync` on every node and verifies `.venv/bin/python` can import `flexllmgen`.

Useful options:
- `HEAD_IP=192.168.79.22`: IP address used by `dist_flex_opt` as the rendezvous head.
- `PORT=7777`: rendezvous port.
- `GPUS_PER_NODE=4`: number of FlexGen ranks to launch per node.
- `CORES_PER_GPU=4`: CPU cores bound to each rank.
- `MPI_PREFIX=/usr`: OpenMPI installation prefix. Use this if remote nodes fail with `orted: command not found`.
- `SSH_PORT=2222`: SSH port used by OpenMPI when every node uses the same non-default port.
- `RUN_SETUP=0`: skip setup validation if the environment is already prepared.
- `COMPRESS_WEIGHT_MODES=on`: run only the `--compress-weight` mode.
- `ALLOW_RUN_AS_ROOT=auto`: automatically adds `--allow-run-as-root` when the script is run as root. Set `ALLOW_RUN_AS_ROOT=0` to disable this.

If nodes use different SSH ports, define aliases in `~/.ssh/config` and use those aliases in `HOSTS` or the hostfile:

```
Host flex-node-1
  HostName 192.168.79.22
  Port 2222
  User root

Host flex-node-2
  HostName 192.168.79.23
  Port 2202
  User root
```

Then run:

```
HOSTS="flex-node-1,flex-node-2" GPUS_PER_NODE=1 ./run_flexgen_dist_benchmark.sh
```

Logs are written to `/logs/flexgen_dist/<run-id>` inside the Docker container, or `./logs/flexgen_dist/<run-id>` outside the container.

If OpenMPI cannot find `orted` on worker nodes, first check that OpenMPI is installed on every node:

```
ssh node-name 'which mpirun; which orted; mpirun --version'
```

If `orted` exists but OpenMPI cannot find it during launch, pass the common install prefix:

```
MPI_PREFIX=/usr HOSTFILE=hosts GPUS_PER_NODE=1 ./run_flexgen_dist_setup.sh
```

```
LD_LIBRARY_PATH= \
HOSTFILE=../../hostfile \
GPUS_PER_NODE=1 \
MPI_PREFIX=/usr \
./run_flexgen_dist_setup.sh

COMPRESS_WEIGHT_MODES=on \
LD_LIBRARY_PATH= \
HOSTFILE=../../hostfile \
GPUS_PER_NODE=1 \
MPI_PREFIX=/usr \
./run_flexgen_dist_benchmark.sh
```

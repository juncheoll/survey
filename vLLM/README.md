### run
```
vllm serve meta-llama/Llama-2-7b-hf \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.90


vllm bench serve \
  --backend openai \
  --model meta-llama/Llama-2-7b-hf \
  --base-url http://0.0.0.0:8000 \
  --endpoint /v1/completions \
  --dataset-name random \
  --num-prompts 64 \
  --random-input-len 1024 \
  --random-output-len 256 \
  --random-range-ratio 0 \
  --request-rate inf \
  --max-concurrency 64
```

### set environment
```
export GLOO_SOCKET_IFNAME=enp7s0
export NCCL_SOCKET_IFNAME=enp7s0
```

### benchmark Llama 70B with Ray
The benchmark wrapper reads host information, starts a Ray cluster, launches `vllm serve`, waits for the OpenAI-compatible API, and runs `vllm bench serve`.

With a hostfile:

```
cp hosts.example hosts
HOSTFILE=hosts ./run_vllm_benchmark.sh
```

Or with a comma-separated host list:

```
HOSTS="192.168.79.22,192.168.79.23" GPUS_PER_NODE=4 ./run_vllm_benchmark.sh
```

Hostfile format:

```
192.168.79.22 slots=4
192.168.79.23 slots=4
```

The first host is treated as the Ray head node. Run the script on that head node. By default, `HEAD_ADDRESS` is detected with `hostname -i`; override it if workers need a different reachable IP:

```
HEAD_ADDRESS=192.168.79.22 HOSTFILE=hosts ./run_vllm_benchmark.sh
```

Defaults:
- `MODEL=meta-llama/Llama-2-70b-hf`
- `TENSOR_PARALLEL_SIZE=auto`: uses the first host's slots.
- `PIPELINE_PARALLEL_SIZE=auto`: uses the number of hosts.
- `RANDOM_INPUT_LEN=1024`
- `RANDOM_OUTPUT_LEN=256`
- `NUM_PROMPTS=64`
- `NUM_PROMPTS_PER_CONCURRENCY=0`
- `MAX_CONCURRENCIES="1 2 4 8"`
- `VLLM_USE_V1=0`
- `MAX_NUM_SEQS` is unset by default, so vLLM uses its own server-side default.

Useful overrides:
- `RUN_RAY_SETUP=0`: reuse an existing Ray cluster.
- `RAY_STOP_FIRST=0`: do not stop existing Ray processes during setup.
- `VLLM_USE_V1=1`: opt back into the vLLM V1 engine.
- `MAX_CONCURRENCIES="16 32 64"`: sweep benchmark concurrency values.
- `NUM_PROMPTS_PER_CONCURRENCY=1`: set `--num-prompts` equal to each `MAX_CONCURRENCIES` value.
- `MAX_NUM_SEQS=64`: explicitly set vLLM server-side max sequences.
- `SSH_PORT=2222`: use the same non-default SSH port for every worker.
- `VLLM_EXTRA_SERVE_ARGS="..."`: append extra args to `vllm serve`.
- `VLLM_EXTRA_BENCH_ARGS="..."`: append extra args to `vllm bench serve`.
- `LOG_DIR=/path/to/logs`: change benchmark log location.

Logs are written to `/logs/vllm/<run-id>` if `/logs` is writable, otherwise `./logs/vllm/<run-id>`.

For single-node multi-GPU pipeline-parallel experiments, use the top-level wrapper. On an 8-GPU node:

```
GPU_COUNTS="8" VLLM_PIPELINE_PARALLEL_SIZE=2 ./run_single_node_multi_gpu_benchmarks.sh
```

This runs vLLM without Ray by default and maps the local 8 GPUs as `TP=4, PP=2`. Override `VLLM_TENSOR_PARALLEL_SIZE` only when you want to choose both values manually; `TP x PP` must equal the GPU count.

For example, this hostfile:

```
192.168.79.22 slots=4
192.168.79.23 slots=4
```

defaults to `--tensor-parallel-size 4` and `--pipeline-parallel-size 2`.

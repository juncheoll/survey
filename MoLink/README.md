### run
```
python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --molink-enabled \
    --molink-grpc-port 50061 \
    --molink-start-layer 0 \
    --molink-end-layer 16 \
    --port 8080 \
    --max-model-len 4096



python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --molink-enabled \
    --molink-grpc-port 50062 \
    --molink-start-layer 16 \
    --molink-end-layer -1 \
    --port 9095 \
    --max-model-len 4096 \
    --molink-initial-peer 192.168.79.9:50061
```

### benchmark Llama 70B with hostfile
MoLink uses pipeline parallelism. This wrapper maps one pipeline stage to one GPU slot, splits the 80 Llama-2-70B layers across stages, starts one MoLink server per stage, and runs `vllm bench serve` against the first stage's OpenAI-compatible API.

Hostfile format:

```
192.168.79.22 slots=1
192.168.79.23 slots=1
```

For single-node multi-GPU, use one host with multiple slots:

```
127.0.0.1 slots=4
```

The wrapper sets `CUDA_VISIBLE_DEVICES` per stage, so `slots=4` starts four local MoLink servers on GPU `0`, `1`, `2`, and `3`.

Run on the first host in the hostfile:

```
cp hosts.example hosts
HEAD_ADDRESS=192.168.79.22 HOSTFILE=hosts ./run_molink_benchmark.sh
```

For two nodes, the default layer split for `meta-llama/Llama-2-70b-hf` is:
- stage 0: layers `0:40`
- stage 1: layers `40:-1`

For four nodes, the split is:
- stage 0: layers `0:20`
- stage 1: layers `20:40`
- stage 2: layers `40:60`
- stage 3: layers `60:-1`

Defaults:
- `MODEL=meta-llama/Llama-2-70b-hf`
- `MODEL_LAYERS=80`
- `MAX_MODEL_LEN=4096`
- `RANDOM_INPUT_LEN=1024`
- `RANDOM_OUTPUT_LEN=256`
- `NUM_PROMPTS=64`
- `NUM_PROMPTS_PER_CONCURRENCY=0`
- `MAX_CONCURRENCIES="1 2 4 8"`
- `IGNORE_EOS=1`

`NUM_PROMPTS` is the total number of requests sent for each benchmark run. `MAX_CONCURRENCIES` is the maximum number of in-flight requests and is the closest knob to request-level batch size for `vllm bench serve`. To measure one wave of exactly `batch_size` requests, set `NUM_PROMPTS_PER_CONCURRENCY=1`; the script will use `--num-prompts` equal to the current `--max-concurrency`.

Useful overrides:
- `SSH_PORT=2222`: use the same non-default SSH port for every worker.
- `REMOTE_MOLINK_DIR=/workspace/MoLink`: path to this directory on every node.
- `GPUS_PER_NODE=4`: when using `HOSTS` instead of `HOSTFILE`, expand each host into four GPU stages.
- `GPU_MEMORY_UTILIZATION=0.75`: pass a lower `--gpu-memory-utilization` to each MoLink server.
- `CLEANUP_EXISTING_SERVERS=0`: skip the pre-launch cleanup of existing MoLink API servers.
- `RANDOM_OUTPUT_LEN=256`: number of generated tokens per request.
- `NUM_PROMPTS=64`: total requests per `MAX_CONCURRENCIES` run.
- `NUM_PROMPTS_PER_CONCURRENCY=1`: set `--num-prompts` equal to each `MAX_CONCURRENCIES` value.
- `MAX_CONCURRENCIES="16 32 64"`: sweep benchmark concurrency values.
- `IGNORE_EOS=0`: omit `--ignore-eos` from `vllm bench serve`.
- `MOLINK_EXTRA_SERVER_ARGS="..."`: append extra args to each MoLink server.
- `VLLM_EXTRA_BENCH_ARGS="..."`: append extra args to `vllm bench serve`.
- `LOG_DIR=/path/to/logs`: change benchmark log location.

Logs are written to `/logs/molink/<run-id>` if `/logs` is writable, otherwise `./logs/molink/<run-id>`.

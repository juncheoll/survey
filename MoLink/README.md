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
MoLink uses pipeline parallelism across single-GPU nodes. The wrapper reads a hostfile, splits the 80 Llama-2-70B layers across nodes, starts one MoLink server per node, and runs `vllm bench serve` against the first node's OpenAI-compatible API.

Hostfile format:

```
192.168.79.22 slots=1
192.168.79.23 slots=1
```

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
- `RANDOM_OUTPUT_LEN=512`
- `NUM_PROMPTS=64`
- `MAX_CONCURRENCIES="1 2 4 8"`
- `IGNORE_EOS=1`

Useful overrides:
- `SSH_PORT=2222`: use the same non-default SSH port for every worker.
- `REMOTE_MOLINK_DIR=/workspace/MoLink`: path to this directory on every node.
- `MAX_CONCURRENCIES="16 32 64"`: sweep benchmark concurrency values.
- `IGNORE_EOS=0`: omit `--ignore-eos` from `vllm bench serve`.
- `MOLINK_EXTRA_SERVER_ARGS="..."`: append extra args to each MoLink server.
- `VLLM_EXTRA_BENCH_ARGS="..."`: append extra args to `vllm bench serve`.
- `LOG_DIR=/path/to/logs`: change benchmark log location.

Logs are written to `/logs/molink/<run-id>` if `/logs` is writable, otherwise `./logs/molink/<run-id>`.

### docker image tagging
```
docker tag zero_inference:cu128 k55369504/zero_inference:cu128
```

### docker image push
```
docker push k55369504/zero_inference:cu128
``` 

### model list
```
facebook/opt-6.7b
facebook/opt-13b
facebook/opt-30b
facebook/opt-66b


meta-llama/Llama-2-7b-hf
meta-llama/Llama-2-13b-hf
huggyllama/llama-30b
meta-llama/Llama-2-70b-hf
```

###
```
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev
```

### single-node benchmark automation
Runs the single-node benchmark scripts for FlexGen, ZeRO-Inference, and SubSpec in sequence. PowerInfer is also supported, but is not included by default because it may download large GGUF model repos.

```
./run_single_node_benchmarks.sh
```

Useful overrides:
- `FRAMEWORKS="FlexGen SubSpec"`: select frameworks.
- `FRAMEWORKS="PowerInfer"`: run only PowerInfer.
- `STOP_ON_FAILURE=1`: stop after the first failed framework.
- `LOG_DIR=/path/to/logs`: change orchestration log location.

The wrapper writes `summary.tsv` and per-framework stdout logs under `logs/single_node/<run-id>` unless `/logs` is writable.

### single-node multi-GPU benchmark automation
Runs the single-node GPU-count sweep for vLLM, FlexGen, and MoLink.

```
GPU_COUNTS="1 2 4" ./run_single_node_multi_gpu_benchmarks.sh
```

Useful overrides:
- `FRAMEWORKS="vLLM MoLink"`: select frameworks.
- `GPU_COUNTS="4 8"`: select local GPU counts.
- `SINGLE_NODE_HOST=127.0.0.1`: host used for local MoLink stages.
- `GPU_MEMORY_UTILIZATION=0.75`: lower vLLM/MoLink GPU memory reservation if startup sees insufficient free memory.
- `VLLM_DISTRIBUTED_EXECUTOR_BACKEND=ray`: force Ray for vLLM; by default single-node runs without Ray.
- `VLLM_USE_V1=0`: force vLLM V0 engine. This is the default for the vLLM benchmark wrapper.
- `VLLM_ENFORCE_EAGER=1`: disable vLLM compile/cudagraph startup path for debugging unstable launches.
- `VLLM_DISABLE_CUSTOM_ALL_REDUCE=1`: explicitly disable vLLM custom all-reduce.
- `STOP_ON_FAILURE=1`: stop after the first failed run.
- `LOG_DIR=/path/to/logs`: change orchestration log location.

For vLLM, the wrapper uses `TP=gpu_count` and `PP=1` without Ray by default. For FlexGen, it runs the distributed benchmark with `GPUS_PER_NODE=gpu_count` on the local node. For MoLink, it expands the local host into `gpu_count` pipeline stages and pins each stage with `CUDA_VISIBLE_DEVICES`.

The wrapper writes `summary.tsv` and per-framework stdout logs under `logs/single_node_multi_gpu/<run-id>` unless `/logs` is writable.

### multi-node benchmark automation
Runs the multi-node benchmark scripts for vLLM, FlexGen, and MoLink in sequence.

```
HOSTFILE=./hostfile HEAD_ADDRESS=192.168.79.22 ./run_multi_node_benchmarks.sh
```

Useful overrides:
- `FRAMEWORKS="vLLM FlexGen"`: select frameworks.
- `STOP_ON_FAILURE=1`: stop after the first failed framework.
- `LOG_DIR=/path/to/logs`: change orchestration log location.
- `VLLM_HOSTFILE=./hosts_4gpu`: hostfile only for vLLM.
- `FLEXGEN_HOSTFILE=./hosts_4gpu`: hostfile only for FlexGen.
- `MOLINK_HOSTFILE=./hosts_molink`: hostfile only for MoLink.

The wrapper passes `HEAD_ADDRESS` and `HEAD_IP` to child scripts. If only one is set, it reuses that value for the other. MoLink maps one pipeline stage to one GPU slot, so a hostfile with `slots=4` starts four MoLink stages on that host.

The wrapper writes `summary.tsv` and per-framework stdout logs under `logs/multi_node/<run-id>` unless `/logs` is writable.








### single-single
```
VRAM_LIMITS="16" ./run_single_node_benchmarks.sh
FRAMEWORKS="PowerInfer" ./run_single_node_benchmarks.sh
```

### single-multi
```
GPU_COUNTS="8" \
NUM_PROMPTS_PER_CONCURRENCY=1 \
VLLM_ENFORCE_EAGER=1 \
./run_single_node_multi_gpu_benchmarks.sh
```

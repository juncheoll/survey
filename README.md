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
Runs the single-node benchmark scripts for FlexGen, ZeRO-Inference, and SubSpec in sequence.

```
./run_single_node_benchmarks.sh
```

Useful overrides:
- `FRAMEWORKS="FlexGen SubSpec"`: select frameworks.
- `STOP_ON_FAILURE=1`: stop after the first failed framework.
- `LOG_DIR=/path/to/logs`: change orchestration log location.

The wrapper writes `summary.tsv` and per-framework stdout logs under `logs/single_node/<run-id>` unless `/logs` is writable.

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
- `MOLINK_HOSTFILE=./hosts_1gpu`: hostfile only for MoLink.

The wrapper passes `HEAD_ADDRESS` and `HEAD_IP` to child scripts. If only one is set, it reuses that value for the other. MoLink is still run as a multi-node pipeline benchmark; this automation maps one pipeline stage to one host, so use a MoLink-specific hostfile with `slots=1` when your vLLM/FlexGen hostfile uses multi-GPU nodes.

The wrapper writes `summary.tsv` and per-framework stdout logs under `logs/multi_node/<run-id>` unless `/logs` is writable.

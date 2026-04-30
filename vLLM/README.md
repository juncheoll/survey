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
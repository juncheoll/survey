### docker build
```
docker build -f Dockerfile -t k55369504/zero_inference:cu128 .
```

### docker run
```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v ./logs:/logs \
  --name zero_inference \
  zero_inference:cu128
```

### create & activate env
```
uv venv /opt/venvs/zero
source /opt/venvs/zero/bin/activate
```

### install ZeRO-Inference
```
git clone https://github.com/deepspeedai/DeepSpeedExamples.git
cd DeepSpeedExamples/inference/huggingface/zero_inference
uv pip install -r requirements.txt
```

### edit python script
```
try:
    from transformers.integrations.deepspeed import HfDeepSpeedConfig
except Exception:
    from transformers.deepspeed import HfDeepSpeedConfig
```

### run ZeRO-Inference
```
deepspeed --num_gpus 1 run_model.py --dummy --model facebook/opt-13b --batch-size 8 --prompt-len 512 --gen-len 32 --cpu-offload --kv-offload

deepspeed --num_gpus 1 run_model.py --dummy --model facebook/opt-66b --batch-size 8 --prompt-len 512 --gen-len 32 --cpu-offload --kv-offload --quant_bits
```

### benchmark Llama models
```
./run_inference_benchmark.sh
```

The script runs `uv sync`, activates `.venv`, and benchmarks:
- `meta-llama/Llama-2-7b-hf`
- `meta-llama/Llama-2-13b-hf`
- `huggyllama/llama-30b`

Each model is tested with batch sizes `1`, `2`, `4`, and `8`, using `--prompt-len 1024` and `--gen-len 512`.

Logs are written to `/logs/zero_inference/<run-id>` inside the Docker container, or `./logs/zero_inference/<run-id>` outside the container.

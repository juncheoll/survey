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
python3 run_model.py --model facebook/opt-13b --batch-size 8 --prompt-len 512 --gen-len 32 --cpu-offload --kv-offload
```
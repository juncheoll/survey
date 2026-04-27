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
python3 -m flexllmgen.flex_opt --model meta-llama/Llama-2-7b-chat-hf --percent 0 100 100 0 100 0
python3 -m flexllmgen.flex_opt --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --percent 0 100 100 0 100 0
```

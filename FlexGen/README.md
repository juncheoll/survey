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

Each model is tested with GPU batch sizes `1`, `2`, `4`, and `8`, using `--prompt-len 1024` and `--gen-len 512`.

Logs are written to `/logs/flexgen/<run-id>` inside the Docker container, or `./logs/flexgen/<run-id>` outside the container.


#### Multi-GPU(Multi-Node)
```
python -m flexllmgen.dist_flex_opt --head-ip 192.168.79.22 --port 7777 --rank 0 --local-rank 0 --world-size 2 --model meta-llama/Llama-2-13b-hf --path _DUMMY_ --percent 80 20 100 0 100 0 --comm-device gpu --async-comm
```

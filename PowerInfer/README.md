### activate venv and install powerinfer
```
source .venv/bin/bactivate
cd PowerInfer
pip install -r requirements.txt
```

### build
```
cmake -S . -B build -DLLAMA_CUBLAS=ON
cmake --build build --config Release
```

### PowerInfer Model List
```
PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF
PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF
```

### Model Download
```
hf download $model
```

### convert
```
python convert.py --outfile /PATH/TO/POWERINFER/GGUF/REPO/MODELNAME.powerinfer.gguf /PATH/TO/ORIGINAL/MODEL /PATH/TO/PREDICTOR
# python convert.py --outfile ./ReluLLaMA-70B-PowerInfer-GGUF/llama-70b-relu.powerinfer.gguf ./SparseLLM/ReluLLaMA-70B ./PowerInfer/ReluLLaMA-70B-Predictor
```

### run
```
./PowerInfer/build/bin/batched-bench $MODEL_PATH 2048 0 9 0
```

### benchmark automation
The benchmark script prepares the uv virtualenv, builds PowerInfer, downloads missing GGUF model repos into `./models`, finds the `.gguf` file, and runs `batched-bench`.

```
./run_powerinfer_benchmark.sh
```

By default it runs:

```
PowerInfer/ReluLLaMA-7B-PowerInfer-GGUF
PowerInfer/ReluLLaMA-13B-PowerInfer-GGUF
PowerInfer/ReluLLaMA-70B-PowerInfer-GGUF
```

Useful overrides:
- `MODEL_CACHE_DIR=/path/to/models`: where Hugging Face repos are downloaded.
- `HF_CLI=hf`: Hugging Face CLI command used for downloads.
- `RUN_BUILD=0`: skip CMake build if `PowerInfer/build/bin/batched-bench` already exists.
- `DOWNLOAD_MODELS=0`: fail instead of downloading missing model repos.
- `VRAM_BUDGET_GB=9`: vram budget passed to `batched-bench`.
- `N_KV_MAX=2048`, `IS_PP_SHARED=0`, `MMQ=0`: remaining default `batched-bench` arguments.
- `PP_LIST="512" TG_LIST="32" PL_LIST="1,2,4,8"`: override `batched-bench` workload lists.

Logs are written to `/logs/powerinfer/<run-id>` if `/logs` is writable, otherwise `./logs/powerinfer/<run-id>`.

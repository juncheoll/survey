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
huggingface-cli download $model
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
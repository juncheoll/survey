### docker build
```
docker build -f Dockerfile -t prima:cu128 .
```

### docker run
```
sudo docker run --gpus all -it --rm \
  --ipc=host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $PWD/download:/download \
  --name prima \
  prima:cu128
```

### download model
```
mkdir download
wget https://huggingface.co/Qwen/QwQ-32B-GGUF/resolve/main/qwq-32b-q4_k_m.gguf -P download/
wget https://huggingface.co/bartowski/opt-1.3b-GGUF/resolve/main/opt-30b.Q4_K_M.gguf -P download/
```


#### install HiGHS from source
```
git clone https://github.com/ERGO-Code/HiGHS.git
cd HiGHS
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
sudo ldconfig
```

#### build prima.cpp
```
cd prima.cpp
make GGML_CUDA=1 USE_HGIHS=1 -j$(npoc)
make GGML_CUDA=1 USE_HIGHS=1 llama-bench -j$(nproc)
```

#### model list
```
TheBloke/Llama-2-7B-GGUF
bartowski/Meta-Llama-3-8B-Instruct-GGUF
RDson/Llama-3-14B-Instruct-v1-GGUF
TheBloke/upstage-llama-30b-instruct-2048-GGUF
```

#### model download
```
hf download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_0.gguf
```


### run PRIMA.cpp
```
./llama-cli -m download/qwq-32b-q4_k_m.gguf -c 1024 -p "what is edge AI?" -n 256 -ngl 30
./llama-cli -m download/opt-1.3b.Q4_K_M.gguf -c 1024 -p "Hello, what is AI?" -n 256 -ngl 30
```


### multi-GPU
```
# rank 0
CUDA_VISIBLE_DEVICES=0 ./llama-server \
  -m model.gguf -c 4096 \
  --world 2 --rank 0 --master 127.0.0.1 --next 127.0.0.1 --prefetch \
  --host 127.0.0.1 --port 8080 \
  -np 4 --cont-batching

# rank 1
CUDA_VISIBLE_DEVICES=1 ./llama-cli \
  -m model.gguf \
  --world 2 --rank 1 --master 127.0.0.1 --next 127.0.0.1 --prefetch


python3 bench_prima_server.py \
  --url http://127.0.0.1:8080 \
  --requests 64 \
  --concurrency 4 \
  --input-tokens 1024 \
  --output-tokens 256

```
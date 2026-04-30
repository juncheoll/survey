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


#### required pacakge
```
sudo apt update
sudo apt install -y fio libzmq3-dev
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
hf download TheBloke/Llama-2-7B-GGUF llama-2-7b.Q4_K_M.gguf
hf download Tobius/llama-2-7b-hf-gguf llama-2-7b-hf.gguf
hf download TheBloke/LLaMA-30b-GGUF llama-30b.Q4_K_S.gguf
bartowski/Meta-Llama-3-70B-Instruct-GGUF Meta-Llama-3-70B-Instruct-Q4_K_M.gguf

hf download KoboldAI/LLaMA2-13B-Psyfighter2-GGUF LLaMA2-13B-Psyfighter2.F16.gguf




hf download meta-llama/Llama-2-70b-hf
```

### convert hf to gguf
```
python3 convert_hf_to_gguf.py models/Llama-2-7b-hf --outtype f16
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
  -m ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/Llama-2-7B-hf-F16.gguf \
  -c 2048 \
  --world 2 --rank 0 --master 192.168.79.22 --next 192.168.79.4 --prefetch \
  --host 127.0.0.1 --port 8080 \
  -np 64 --cont-batching \
  -lw "40,20" -ngl 40

# rank 1
CUDA_VISIBLE_DEVICES=0 ./llama-cli \
  -m ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9/Llama-2-7B-hf-F16.gguf \
  --world 2 --rank 1 --master 192.168.79.22 --next 192.168.79.22 --prefetch \
  -ngl 20


python3 bench_prima_server.py \
  --url http://127.0.0.1:8080 \
  --requests 64 \
  --concurrency 64 \
  --input-tokens 1024 \
  --output-tokens 256

```
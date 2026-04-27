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

### run PRIMA.cpp
```
./llama-cli -m download/qwq-32b-q4_k_m.gguf -c 1024 -p "what is edge AI?" -n 256 -ngl 30
./llama-cli -m download/opt-1.3b.Q4_K_M.gguf -c 1024 -p "Hello, what is AI?" -n 256 -ngl 30
```
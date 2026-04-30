### run
```
python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --molink-enabled \
    --molink-grpc-port 50061 \
    --molink-start-layer 0 \
    --molink-end-layer 16 \
    --port 8080 \
    --max-model-len 4096



python -m molinkv1.entrypoints.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --molink-enabled \
    --molink-grpc-port 50062 \
    --molink-start-layer 16 \
    --molink-end-layer -1 \
    --port 9095 \
    --max-model-len 4096 \
    --molink-initial-peer 192.168.79.9:50061
```
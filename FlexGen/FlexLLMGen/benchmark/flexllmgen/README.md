# Benchmark FlexLLMGen
NOTE: This benchmark uses dummy weights by default for faster experiments.
It is expected if you see randomly generated garbled characters, but the throughput and latency numbers should be correct.

## Mount SSD
The following commands use `~/flexllmgen_offload_dir` as the offloading folder by default.
To get the best performance, it is recommonded to mount this folder on a fast SSD.
If you use AWS or GCP instances with local SSDs, you can use [mount_nvme_aws.sh](../../scripts/mount_nvme_aws.sh) or [mount_nvme_gcp.sh](../../scripts/mount_nvme_gcp.sh) to mount the local SSDs.

## Single GPU

### OPT-6.7B
```
# fp16
python3 bench_suite.py 6b7_1x1

# with int4 compression
python3 bench_suite.py 6b7_1x1_comp
```

### OPT-30B
```
# fp16
python3 bench_suite.py 30b_1x1

# with int4 compression
python3 bench_suite.py 30b_1x1_comp
```

### OPT-175B
```
# fp16
python3 bench_suite.py 175b_1x1

# with int4 compression
python3 bench_suite.py 175b_1x1_comp
```

## Distributed GPUs

### Requirements
```
sudo apt install openmpi-bin
```

### Environment Setup

**Important**: Activate your virtual environment BEFORE running any benchmark scripts:

```bash
# For uv venv
source .venv/bin/activate

# For conda
conda activate flexllmgen
```

The scripts will automatically detect and use the activated Python environment. Without activation, scripts may use system Python or `~/.local` packages, leading to incorrect behavior.

### Multi-Node Setup (Required for 4x1 benchmarks)

**Important: Run benchmark scripts ONLY on the HEAD NODE. MPI will automatically distribute work to all worker nodes.**

The scripts require a running Ray cluster. You have two options:

**Option 1: Use RAY_ADDRESS environment variable (Recommended - No YAML needed)**

```bash
# On head node, start Ray
ray start --head --port=6379

# On each worker node, join the cluster
ray start --address='HEAD_IP:6379'

# Then run benchmark ONLY on head node with RAY_ADDRESS environment variable
export RAY_ADDRESS=HEAD_IP:6379
./bench_llama_dist_multi_node.sh  # <-- RUN ON HEAD NODE ONLY
```

**Option 2: Use bootstrap config file** (Optional)

Create `~/ray_bootstrap_config.yaml`:
```yaml
cluster_name: flexllmgen
provider:
  type: local
```

Then run on head node:
```bash
./bench_llama_dist_multi_node.sh  # <-- RUN ON HEAD NODE ONLY
```

**Verify cluster is running**:
```bash
ray status
```

### OPT-6.7B
```
# 1 node with 4 GPUs
bash bench_6.7b_1x4.sh

# 4 nodes and one GPU per node
bash bench_6.7b_4x1.sh
```

### OPT-30B
```
# 1 node with 4 GPUs
bash bench_30b_1x4.sh

# 4 nodes and one GPU per node
bash bench_30b_4x1.sh
```

### OPT-175B
```
# 1 node with 4 GPUs
bash bench_175b_1x4.sh

# 4 nodes and one GPU per node
bash bench_175b_4x1.sh
```

### Llama-7B
```
# 1 node with 4 GPUs
bash bench_llama_dist_single_node.sh

# 4 nodes and one GPU per node
bash bench_llama_dist_multi_node.sh
```

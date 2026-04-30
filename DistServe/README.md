### install
```bash
# install DistServe
uv sync

# build SwiftTransformer with the uv environment
bash scripts/build_swifttransformer_uv.sh

# install DistServe
uv pip install -e ./DistServe
```

`scripts/build_swifttransformer_uv.sh` is path-portable. It uses this checkout's
`.venv/bin/python` and lets CMake discover NCCL from the `nvidia-nccl-cu12`
wheel installed with PyTorch. If your GPU is not A100/Ampere-or-newer, override
the build architectures, for example:

```bash
TORCH_CUDA_ARCH_LIST="8.9;9.0" bash scripts/build_swifttransformer_uv.sh
```

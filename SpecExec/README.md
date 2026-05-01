```
cd SpecExec/specexec

../.venv/bin/python run_exp.py \
  --top_p 0.9 \
  --temperature 0.6 \
  --gen_type SpecExecBase \
  --max_budget 128 \
  --test-input-tokens 1024 \
  --max-new-tokens 256 \
  --ignore-eos \
  --n_tests 1 \
  --offload \
  --exp_name SX_sample_offload_1024_256
```



```
cd SpecExec/specexec

../.venv/bin/python run_exp.py \
  --model_0 TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --model_1 meta-llama/Llama-2-13b-chat-hf \
  --gen_type SpecExecBase \
  --test-input-tokens 1024 \
  --max-new-tokens 256 \
  --ignore-eos \
  --n_tests 1 \
  --offload \
  --exp_name SX_llama13b_1024_256
```
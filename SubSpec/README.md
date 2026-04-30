### run
```
python -m run.main \
  --config configs/exp_offloading/subspec_sd_llama_7b_vram_10gb.yaml \
  --test-input-tokens 1024 \
  --max-new-tokens 128 \
  --ignore-eos \
  run-test
```
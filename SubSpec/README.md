### run
```
python -m run.main \
  --config configs/exp_offloading/subspec_sd_llama_7b_vram_10gb.yaml \
  --test-input-tokens 1024 \
  --max-new-tokens 256 \
  --ignore-eos \
  run-test

SUBSPEC_HQQ_BACKEND=pytorch \
python -m run.main \
  --config configs/exp_offloading/subspec_sd_llama_7b_vram_16gb.yaml \
  --test-input-tokens 1024 \
  --max-new-tokens 256 \
  --ignore-eos \
  --compile-mode none \
  --warmup-iter 1 \
  run-test

```

### benchmark offloading configs
```
./run_subspec_benchmark.sh
```

The script runs `uv sync`, activates `.venv`, and runs all model sizes:
- `7b`
- `13b`
- `30b`

By default, each model is tested with VRAM limits `10`, `12`, and `16` GB, using `--test-input-tokens 1024`, `--max-new-tokens 256`, and `--ignore-eos`.

Select VRAM limits by passing them as arguments:

```
VRAM_LIMITS="14 18" SUBSPEC_SAFE_MODE=1 ./SubSpec/run_subspec_benchmark.sh
```

Useful overrides:
- `MODEL_SIZES="7b 13b"`: run only selected model sizes.
- `TEST_INPUT_TOKENS=2048`: override input token count.
- `MAX_NEW_TOKENS=256`: override output token count.
- `IGNORE_EOS=0`: omit `--ignore-eos`.
- `LOG_DIR=/path/to/logs`: change benchmark stdout/summary location.

SubSpec's own run logs are still written under `subspec/experiments`. The wrapper writes stdout logs and `summary.tsv` under `logs/subspec_benchmark/<run-id>`.

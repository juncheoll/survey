import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import trange
import logging
import os
import nvtx
import random
import time
import json
from specdecodes.models.utils.wandb_logger import wandb_logger
from run.pipelines.utils.eval_utils import reset_kv, maybe_init_cuda_graph_runner
from run.core.config_utils import write_settings_yaml


def _build_input_ids(tokenizer, messages, device):
    try:
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(device)
    except ValueError as exc:
        if "chat_template" not in str(exc):
            raise

    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    parts.append("assistant:")
    prompt = "\n".join(parts)
    return tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)


def _make_length_controlled_prompt(tokenizer, prompt_text, target_tokens):
    token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if not token_ids:
        token_ids = tokenizer.encode(" ", add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Could not tokenize test prompt for {target_tokens} input tokens")

    separator = "\n\n"
    repeat_ids = tokenizer.encode(separator + prompt_text, add_special_tokens=False)
    if not repeat_ids:
        repeat_ids = token_ids

    input_ids = list(token_ids)
    while len(input_ids) < target_tokens:
        input_ids.extend(repeat_ids)

    input_ids = input_ids[:target_tokens]
    prompt = tokenizer.decode(input_ids, skip_special_tokens=False)
    return prompt, torch.tensor([input_ids], dtype=torch.long)


def _prepare_test_input(tokenizer, args, input_message):
    n_input_tokens = getattr(args, "test_input_tokens", None)
    if n_input_tokens is not None:
        n_input_tokens = int(n_input_tokens)
        if n_input_tokens <= 0:
            raise ValueError("test_input_tokens must be a positive integer")
        prompt_source = getattr(args, "test_input_text", None) or input_message
        prompt, input_ids = _make_length_controlled_prompt(tokenizer, prompt_source, n_input_tokens)
        return input_message, prompt, input_ids.to(args.device), None

    messages = [{"role": "user", "content": input_message}]
    tokenizer.use_default_system_prompt = True
    input_ids = _build_input_ids(tokenizer, messages, args.device)
    prompt = tokenizer.decode(input_ids[0])
    return input_message, prompt, input_ids, None


def _generate_kwargs(args, input_ids, past_kv, draft_past_kv):
    kwargs = {
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "past_key_values": past_kv,
        "draft_past_key_values": draft_past_kv,
        "ignore_eos": getattr(args, "ignore_eos", False),
    }
    max_new_tokens = getattr(args, "max_new_tokens", None)
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = int(max_new_tokens)
    else:
        kwargs["max_length"] = args.max_length
    return kwargs


def _token_count(token_ids):
    if hasattr(token_ids, "numel"):
        return int(token_ids.numel())
    try:
        return len(token_ids)
    except TypeError:
        return 1


def main(builder):
    generator, tokenizer, past_kv, draft_past_kv = builder.build()
    args = builder.args
    
    # set logging level by environment variable
    LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(level=LOGLEVEL)

    # deterministic
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    # warm up
    if args.warmup_iter > 0:
        print("Warming up... It will take some time for the first few iterations to run.")
        with nvtx.annotate("Warming up"):
            is_profiling = generator.profiling
            generator.profiling = False
            for i in trange(args.warmup_iter, desc='Warming up'):
                warmup_message = "Write an essay about large language models."
                with nvtx.annotate("Warm up"):
                    _, _, input_ids, _ = _prepare_test_input(tokenizer, args, warmup_message)
                    with sdpa_kernel(backends=[SDPBackend.MATH]):
                        generator.generate(
                            input_ids,
                            **_generate_kwargs(args, input_ids, past_kv, draft_past_kv),
                        )
                
                reset_kv(past_kv, draft_past_kv)
            generator.profiling = is_profiling

    # Optional CUDA-graph capture for FlashInfer, after warmup (stabilizes kernels/allocations).
    maybe_init_cuda_graph_runner(generator, past_kv, draft_past_kv, args.device, args.warmup_iter)
        
    # input message
    input_message = (
        args.test_prompt
        or "Do you know what is Beyblade? What is the best strategy to build the strongest Beyblade?"
    )
    input_message, prompt, input_ids, synthetic_token_id = _prepare_test_input(tokenizer, args, input_message)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    token_timestamps_s = []
    token_counts = []
    wall_start_s = None

    def stream_callback(token_ids):
        if getattr(args, "sync_token_timing", True):
            torch.cuda.synchronize()
        token_timestamps_s.append(time.perf_counter() - wall_start_s)
        token_counts.append(_token_count(token_ids))
                  
    # generate response
    print("Generating response...")
    torch.cuda.cudart().cudaProfilerStart() # start profiling from here
    wall_start_s = time.perf_counter()
    start_event.record()
    with nvtx.annotate("Generate"):
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            output_ids = generator.generate(
                input_ids,
                **_generate_kwargs(args, input_ids, past_kv, draft_past_kv),
                stream_callback=stream_callback,
            )
    end_event.record()
    
    # Ensure all CUDA kernels are done.
    torch.cuda.synchronize()
    wall_total_s = time.perf_counter() - wall_start_s
    torch.cuda.cudart().cudaProfilerStop()
    
    total_time_s = start_event.elapsed_time(end_event) / 1000.0
    output = generator.tokenizer.decode(output_ids[0][input_ids.shape[1]:])
    n_output_tokens = int(output_ids.shape[1] - input_ids.shape[1])
    ttft_s = token_timestamps_s[0] if token_timestamps_s else None
    tpot_s = None
    if ttft_s is not None and n_output_tokens > 1:
        tpot_s = (wall_total_s - ttft_s) / (n_output_tokens - 1)

    # Persist a single-run log (mirrors benchmark JSONL style).
    log_dir = os.path.join(args.log_dir, time.strftime("%Y%m%d-%H%M%S"), "run_test")
    os.makedirs(log_dir, exist_ok=True)
    write_settings_yaml(log_dir, getattr(args, "settings_snapshot", None))
    log_file = os.path.join(log_dir, "0.jsonl")
    exp_log = {
        **wandb_logger.log_data,
        "input_message": input_message,
        "prompt": prompt,
        "response": output,
        "synthetic_input": synthetic_token_id is not None,
        "synthetic_input_token_id": synthetic_token_id,
        "test_input_tokens": getattr(args, "test_input_tokens", None),
        "test_input_text": getattr(args, "test_input_text", None),
        "test_input_token_text": getattr(args, "test_input_token_text", None),
        "ignore_eos": bool(getattr(args, "ignore_eos", False)),
        "elapsed_time": float(total_time_s),
        "elapsed_time_wall": float(wall_total_s),
        "ttft": float(ttft_s) if ttft_s is not None else None,
        "ttft_ms": float(ttft_s * 1000.0) if ttft_s is not None else None,
        "tpot": float(tpot_s) if tpot_s is not None else None,
        "tpot_ms": float(tpot_s * 1000.0) if tpot_s is not None else None,
        "token_timestamps": [float(ts) for ts in token_timestamps_s],
        "stream_token_counts": token_counts,
        "n_stream_callbacks": len(token_timestamps_s),
        "n_prompt_tokens": int(input_ids.shape[1]),
        "n_output_tokens": n_output_tokens,
        "peak_memory": float(torch.cuda.max_memory_reserved(args.device) / (1024**3)),
    }
    with open(log_file, "a+", encoding="utf-8") as f:
        json.dump(exp_log, f, indent=4)
        f.write("\n")
    print(f"Log directory: {log_dir}")

    if args.print_message:
        print("\nPrompt:")
        print(prompt)
        print("\nModel response:")
        print(output)
        print("\n-----------------------------------")
        print("Input tokens:", len(input_ids[0]))
        print("Output tokens:", len(output_ids[0][input_ids.shape[1]:]))
    
    if args.print_time:
        print("Time:", total_time_s)
        if ttft_s is not None:
            print("TTFT:", ttft_s)
        if tpot_s is not None:
            print("TPOT:", tpot_s)

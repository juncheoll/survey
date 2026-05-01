import torch
import time
import os
import json
from transformers import AutoTokenizer

from EasySpec.models.distributed import DistributedInferenceEngine
from EasySpec.generation.utils.candidate_generator import CandidateRefillPolicy
from EasySpec.models.hyperdraft.hyperdraft_info import LayerParallelPolicy
# from model_utils import stream_jsonl, write_jsonl
from EasySpec.dataset.get_data_prompts import get_prompts_from_name
from EasySpec.utils import print_and_record
import datetime
from pathlib import Path


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)


def _env_list(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _load_prompt_text(prompts_file):
    with open(prompts_file, "r") as f:
        data = json.load(f)
    if not data:
        raise ValueError(f"No prompts found in {prompts_file}")
    first = data[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2:
        return first[1]
    if isinstance(first, dict):
        for key in ("prompt", "text", "input", "question"):
            if key in first:
                return first[key]
    if isinstance(first, str):
        return first
    raise ValueError(f"Unsupported prompt format in {prompts_file}")


def _make_length_controlled_prompt(text, tokenizer, target_tokens):
    if target_tokens is None or target_tokens <= 0:
        return text

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Cannot build length-controlled prompt from empty text")

    if len(token_ids) >= target_tokens:
        controlled_ids = token_ids[:target_tokens]
    else:
        repeats = (target_tokens + len(token_ids) - 1) // len(token_ids)
        controlled_ids = (token_ids * repeats)[:target_tokens]

    return tokenizer.decode(controlled_ids, skip_special_tokens=False)


def _default_specexec_prompts_file():
    return str(
        Path(__file__).resolve().parents[2]
        / "SpecExec"
        / "specexec"
        / "data"
        / "oasst_prompts.json"
    )


def _find_cached_snapshot(model_id):
    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model_cache = cache_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = model_cache / "snapshots"
    if not snapshots_dir.is_dir():
        return None

    snapshots = [
        path for path in snapshots_dir.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    ]
    if not snapshots:
        return None
    return str(max(snapshots, key=lambda path: path.stat().st_mtime))


def get_model_path(model_name):
    if os.path.exists(model_name):
        return model_name

    model_root_dir = os.environ.get("MODEL_ROOT_DIR")
    if model_root_dir:
        model_path = os.path.join(model_root_dir, model_name)
        if os.path.exists(model_path):
            return model_path

    cached_snapshot = _find_cached_snapshot(model_name)
    if cached_snapshot is not None:
        return cached_snapshot

    return model_name


def _assistant_bound_strategy(assistant_model_path):
    assistant_model_path = str(assistant_model_path)

    llama3_tokenizer_kwargs = {"tokenize": False, "add_generation_prompt": False}
    qwen2_tokenizer_kwargs = {"tokenize": False, "add_generation_prompt": False}

    if "Meta-Llama-3-8B-Instruct" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26,27], [28,29,30], [31]]
        tokenizer_kwargs = llama3_tokenizer_kwargs
    elif "Llama-2-7b" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26,27], [28,29,30], [31]]
        tokenizer_kwargs = llama3_tokenizer_kwargs
    elif "Llama-2-13b" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26,27], [28,29,30,31], [32,33,34,35], [36,37,38], [39]]
        tokenizer_kwargs = llama3_tokenizer_kwargs
    elif "Llama-3.2-3B-Instruct" in assistant_model_path:
        bound_strategy = [[0], [1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16], [17,18], [19,20], [21,22],
                          [23,24], [25,26], [27]]
        tokenizer_kwargs = llama3_tokenizer_kwargs
    elif "Llama-3.2-1B-Instruct" in assistant_model_path:
        bound_strategy = [[0], [1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15]]
        tokenizer_kwargs = llama3_tokenizer_kwargs
    elif "TinyLlama" in assistant_model_path:
        bound_strategy = [[0], [1,2], [3,4], [5,6], [7,8], [9,10], [11,12], [13,14], [15,16], [17,18], [19,20], [21]]
        tokenizer_kwargs = llama3_tokenizer_kwargs
    elif "Qwen2-7B" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26], [27]]
        tokenizer_kwargs = qwen2_tokenizer_kwargs
    elif "Qwen2-1.5B" in assistant_model_path:
        bound_strategy = [[0], [1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14], [15,16,17], [18,19,20], [21,22,23],
                          [24,25,26], [27]]
        tokenizer_kwargs = qwen2_tokenizer_kwargs
    elif "Qwen2-0.5B" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22], [23]]
        tokenizer_kwargs = qwen2_tokenizer_kwargs
    elif "Qwen2.5-7B-Instruct" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26,27]]
        tokenizer_kwargs = qwen2_tokenizer_kwargs
    elif "Qwen2-Math-7B-Instruct" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26], [27]]
        tokenizer_kwargs = qwen2_tokenizer_kwargs
    elif "Qwen2.5-Coder-7B-Instruct" in assistant_model_path:
        bound_strategy = [[0], [1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19], [20,21,22,23],
                          [24,25,26], [27]]
        tokenizer_kwargs = qwen2_tokenizer_kwargs
    else:
        raise ValueError(f"Unsupported assistant model for bound_strategy: {assistant_model_path}")

    return bound_strategy, tokenizer_kwargs
    

def main():
    show_output = _env_bool("EASYSPEC_SHOW_OUTPUT", False)
    show_output_ids = _env_bool("EASYSPEC_SHOW_OUTPUT_IDS", False)
    record_in_file = _env_bool("EASYSPEC_RECORD_IN_FILE", True)

    visible_devices = os.environ.get("EASYSPEC_CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7"))
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_devices
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ["NCCL_DEBUG"] = os.environ.get("EASYSPEC_NCCL_DEBUG", "WARN")
    world_size = _env_int("EASYSPEC_WORLD_SIZE", len([device for device in visible_devices.split(",") if device.strip()]))
    model_dtype = torch.bfloat16
                
    candidate_refill_policy = CandidateRefillPolicy.EVERY_ROUND
    # candidate_refill_policy = CandidateRefillPolicy.PREFILL_ONLY
    layer_parallel_policy = LayerParallelPolicy.ATTN_ONLY

    
    test_input_tokens = _env_int("EASYSPEC_TEST_INPUT_TOKENS", 0)
    test_input_text = os.environ.get("EASYSPEC_TEST_INPUT_TEXT")
    prompts_file = os.environ.get("EASYSPEC_PROMPTS_FILE", _default_specexec_prompts_file())
    use_specexec_prompt = _env_bool("EASYSPEC_USE_SPECEXEC_PROMPT", test_input_tokens > 0 or test_input_text is not None)
    ignore_eos = _env_bool("EASYSPEC_IGNORE_EOS", False)
    default_datasets = ["specexec_prompt"] if use_specexec_prompt else ['mmlu', 'humaneval', 'math', 'ifeval_strict', 'mgsm']
    datasets = _env_list("EASYSPEC_DATASETS", default_datasets)
    tree_hyper_params = [(
        _env_int("EASYSPEC_TREE_DEPTH", 5),
        _env_int("EASYSPEC_TOP_K", 12),
    )]

    base_model_name = os.environ.get("EASYSPEC_BASE_MODEL", "meta-llama/Llama-2-70b-chat-hf")
    assistant_model_name = os.environ.get("EASYSPEC_ASSISTANT_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    model_paths = [
        (get_model_path(base_model_name), get_model_path(assistant_model_name)),
    ]
    
    for model_dir, assistant_model_path in model_paths:
        for depth, top_k in tree_hyper_params:
            for dataset_name in datasets:
                test_start = _env_int("EASYSPEC_TEST_START", 0)
                test_end = _env_int("EASYSPEC_TEST_END", 80)
                num_new_tokens = _env_int("EASYSPEC_NUM_NEW_TOKENS", 1024)
                assistant_confidence_threshold = None
                num_assistant_tokens = 5
                do_sample = _env_bool("EASYSPEC_DO_SAMPLE", True)
                temperature = _env_float("EASYSPEC_TEMPERATURE", 0.8)
                use_assistant_model = _env_bool("EASYSPEC_USE_ASSISTANT_MODEL", True)
                enable_tree_attention = _env_bool("EASYSPEC_ENABLE_TREE_ATTENTION", True)
                if enable_tree_attention:
                    num_assistant_tokens = depth
                                
                if assistant_model_path is not None:
                    bound_strategy, tokenizer_kwargs = _assistant_bound_strategy(assistant_model_path)
                else:
                    bound_strategy = []
                    tokenizer_kwargs = {"tokenize": False, "add_generation_prompt": False}
                
                from_pretrained_kwargs = {
                    "world_size": world_size,
                    "use_tp": True,
                    "use_hyperdraft": False,
                    "torch_dtype": model_dtype,
                }    
                from_pretrained_kwargs["assistant_model_path"] = assistant_model_path
                assistant_from_pretrained_kwargs = from_pretrained_kwargs.copy()
                assistant_from_pretrained_kwargs.pop("assistant_model_path")
                # run layer-parallel drafter with this config:
                assistant_from_pretrained_kwargs.update(
                    {
                        "use_tp": False,
                        "use_hyperdraft": True,
                        "bound_strategy": bound_strategy,
                        "layer_parallel_policy": layer_parallel_policy,
                        "torch_dtype": model_dtype,
                    }
                )
                
                # run forced-TP drafter with this config:
                # assistant_from_pretrained_kwargs.update(
                #     {
                #         "world_size": world_size,
                #         "use_tp": True,
                #         "use_hyperdraft": False,
                #         "torch_dtype": model_dtype,
                #     }
                # )
                
                if assistant_model_path is not None:
                    from_pretrained_kwargs.update({"assistant_from_pretrained_kwargs": assistant_from_pretrained_kwargs})
                                
                sample_str = f"sample{temperature}" if do_sample else "nosample"
                _model_name_path = os.path.basename(assistant_model_path)
                # print(assistant_model_path, _model_name_path)
                _result_dir = os.path.join(os.path.dirname(__file__), f"easyspec_res/{_model_name_path}")
                if use_assistant_model is False:
                    assert candidate_refill_policy == CandidateRefillPolicy.PREFILL_ONLY
                    result_dir = os.path.join(_result_dir, f"results_pureTP{world_size}")
                elif top_k == 1:
                    assert candidate_refill_policy == CandidateRefillPolicy.PREFILL_ONLY
                    assert max(len(b) for b in bound_strategy) == 1
                    assert enable_tree_attention is True
                    result_dir = os.path.join(_result_dir, f"results_plus_sd")
                elif max(len(b) for b in bound_strategy) == 1:
                    assert candidate_refill_policy == CandidateRefillPolicy.PREFILL_ONLY
                    assert enable_tree_attention is True
                    result_dir = os.path.join(_result_dir, f"results_plus_tree")
                else:    
                    assert candidate_refill_policy == CandidateRefillPolicy.EVERY_ROUND
                    assert enable_tree_attention is True
                    result_dir = os.path.join(_result_dir, f"results_{depth}_{top_k}")
                
                result_dir = os.path.join(result_dir, f"{sample_str}", dataset_name)
                print(result_dir)
                affix = model_dir.split("/")[-1]
                if not os.path.isdir(result_dir):
                    os.makedirs(result_dir)
                else:
                    # if already has result, just continue
                    already_has_result = False
                    for file in os.listdir(result_dir):
                        if affix in file:
                            print(f"result of {affix} in {result_dir} already exists.")
                            already_has_result = True
                            break
                    if already_has_result and record_in_file:
                        continue
                
                tokenizer = AutoTokenizer.from_pretrained(model_dir)
                if test_input_text is None and use_specexec_prompt:
                    test_input_text = _load_prompt_text(prompts_file)
                if test_input_text is not None:
                    prompt = _make_length_controlled_prompt(test_input_text, tokenizer, test_input_tokens)
                    prompts = [prompt]
                    test_start = 0
                    test_end = 1
                else:
                    prompts = get_prompts_from_name(dataset_name, use_generator=False, tokenizer=tokenizer, tokenizer_kwargs=tokenizer_kwargs)
                model = DistributedInferenceEngine.from_pretrained(
                    model_dir, 
                    **from_pretrained_kwargs
                )
                model.cache_type = "Dynamic"
                
                now = datetime.datetime.now()
                time_str = now.strftime("%Y-%m-%d_%H-%M-%S")
                # get_result_name
                result_file_name = os.path.join(result_dir, f"result_depth{depth}topk{top_k}_{affix}_{time_str}.txt")
                if layer_parallel_policy == LayerParallelPolicy.DIRECT_MLP:
                    max_parallel_length = max(len(b) for b in bound_strategy)
                    result_file_name = os.path.join(result_dir, f"result_MLP{max_parallel_length}_depth{depth}topk{top_k}_{affix}_{time_str}.txt")
                if record_in_file:
                    fd = open(result_file_name, "w")
                else:
                    fd = None
                
                print_and_record(fd, f"model_dir: {model_dir}")
                print_and_record(fd, f"assistant_model_dir: {assistant_model_path}")
                print_and_record(fd, f"dataset_name: {dataset_name}")
                print_and_record(fd, f"test_start: {test_start}")
                print_and_record(fd, f"test_end: {test_end}")
                print_and_record(fd, f"candidate_refill_policy: {candidate_refill_policy}")
                print_and_record(fd, f"layer_parallel_policy: {layer_parallel_policy}")
                print_and_record(fd, f"num_assistant_tokens: {num_assistant_tokens}")
                print_and_record(fd, f"num_new_tokens: {num_new_tokens}")
                print_and_record(fd, f"test_input_tokens: {test_input_tokens}")
                print_and_record(fd, f"test_input_source: {'custom/spec_exec' if test_input_text is not None else dataset_name}")
                print_and_record(fd, f"ignore_eos: {ignore_eos}")
                print_and_record(fd, f"assistant_confidence_threshold: {assistant_confidence_threshold}")
                print_and_record(fd, f"use_assistant_model: {use_assistant_model}")
                print_and_record(fd, f"enable_tree_attention: {enable_tree_attention}")
                print_and_record(fd, f"depth: {depth}")
                print_and_record(fd, f"top_k: {top_k}")
                print_and_record(fd, f"do_sample: {do_sample}")
                print_and_record(fd, f"temperature: {temperature}")
                print_and_record(fd, f"world_size: {world_size}")
                print_and_record(fd, f"bound_strategy: {bound_strategy}")
                

                torch.cuda.current_stream().synchronize()
                start = time.time()
                
                total_gen_length = 0
                total_ttft = 0.0
                ttft_count = 0
                for i,prompt in enumerate(prompts):
                    if i < test_start:
                        continue                    
                    input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to('cuda')
                    input_length = input_ids.shape[-1]
                    now_raw_time = time.time()
                    print_and_record(fd, f"test item: {i} time {now_raw_time - start}")
                    res_dict = model.generate([prompt], 
                                            do_sample=do_sample,
                                            temperature=temperature,
                                            # top_p=1.0,
                                            max_new_tokens=num_new_tokens,
                                            pad_token_id = 2,
                                            use_assistant_model=use_assistant_model,
                                            num_assistant_tokens=num_assistant_tokens,
                                            candidate_refill_policy=candidate_refill_policy,
                                            assistant_confidence_threshold=assistant_confidence_threshold,
                                            enable_tree_attention=enable_tree_attention,
                                            depth=depth,
                                            top_k=top_k,
                                            eos_token_id=None if ignore_eos else tokenizer.eos_token_id,
                                            )
                    output_ids = res_dict["output_ids"]
                    run_time = res_dict["run_time"]
                    this_run_time = res_dict["this_run_time"]
                    this_num_accepted_tokens = res_dict["this_num_accepted_tokens"]
                    this_num_candidate_tokens = res_dict["this_num_candidate_tokens"]
                    this_assistant_runtime = res_dict["this_assistant_runtime"]
                    this_ttft = res_dict.get("this_ttft", 0.0)
                    if this_ttft > 0:
                        total_ttft += this_ttft
                        ttft_count += 1
                    num_accepted_tokens = res_dict["num_accepted_tokens"]
                    num_candidate_tokens = res_dict["num_candidate_tokens"]
                    assistant_runtime = res_dict["assistant_runtime"]
                    
                    this_gen_length = output_ids[0].shape[-1] - input_length
                    total_gen_length += this_gen_length
                    
                    if show_output_ids:
                        print(output_ids[0,input_length:])
                    if show_output:
                        print(tokenizer.decode(output_ids[0], skip_special_tokens=False))
                        
                    # record statistics
                    print_and_record(fd, f"generated length: {this_gen_length}, total length: {total_gen_length}")
                    print_and_record(fd, f"this ttft: {this_ttft}")
                    print_and_record(fd, f"this token/s ****: {this_gen_length / this_run_time}")
                    print_and_record(fd, f"this accept: {this_num_accepted_tokens}/{this_num_candidate_tokens}={this_num_accepted_tokens / (this_num_candidate_tokens + 1e-5)}")
                    print_and_record(fd, f"full accept: {num_accepted_tokens}/{num_candidate_tokens}={num_accepted_tokens / (num_candidate_tokens + 1e-5)}")
                        
                    if (i - test_start) % 40 == 39 and fd is not None:
                        fd.flush()
                        
                    if i >= test_end - 1:
                        break
                    
                torch.cuda.current_stream().synchronize()
                period = time.time() - start
                
                if isinstance(model, DistributedInferenceEngine):
                    print_and_record(fd, f"manager full time: {period}")
                    worker_run_time, sum_generation_tokens, accept_num, candidate_num, assistant_model_run_time = model.get_generation_time()
                    model.destroy()
                
                # print(tokenizer.decode(output_ids[0], skip_special_tokens=True))

                if sum_generation_tokens != total_gen_length:
                    print(f"sum_generation_tokens: {sum_generation_tokens}, total gen length: {total_gen_length}")
                
                exact_run_time = run_time
                print_and_record(fd, f"exact run time: {exact_run_time}, worker full time: {worker_run_time}")
                print_and_record(fd, f"assistant_model_run_time: {assistant_model_run_time}")
                print_and_record(fd, f"sum_generation_tokens: {sum_generation_tokens}")
                print_and_record(fd, f"tokens per second: {sum_generation_tokens/exact_run_time}")
                print_and_record(fd, f"ttft: {total_ttft / max(ttft_count, 1)}")
                print_and_record(fd, f"tpot: {exact_run_time / (sum_generation_tokens + 1e-5)}")
                print_and_record(fd, f"accept_rate: {accept_num}/{candidate_num} = {accept_num / (candidate_num+1e-5)}")
                
                if fd is not None:
                    fd.close()
        

if __name__ == "__main__":
    main()

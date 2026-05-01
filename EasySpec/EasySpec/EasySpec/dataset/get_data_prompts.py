import json
import os
from .load_llama3eval import get_llama3_eval_prompts
from .load_mgsm import get_mgsm_prompts
from .load_specbench import get_specbench_prompts
from .load_mmlu import get_mmlu_prompts
from .load_mt_bench import get_mt_bench_prompts

NAME2DSFN = {
    "mmlu": get_mmlu_prompts,
    "humaneval": get_llama3_eval_prompts,
    "math": get_llama3_eval_prompts,
    "ifeval_strict": get_llama3_eval_prompts,
    "mgsm": get_mgsm_prompts,
    "math_hard": get_llama3_eval_prompts,
    "mbpp": get_llama3_eval_prompts,
    
    "mt_bench": get_mt_bench_prompts,
    "translation": get_specbench_prompts,
    "summarization": get_specbench_prompts,
    "qa": get_specbench_prompts,
    "math_reasoning": get_specbench_prompts,
    "rag": get_specbench_prompts,
}

def load_dataset_config():
    config_file_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if not os.path.isfile(config_file_path):
        raise ValueError(f"{config_file_path} does not exist.")
    with open(config_file_path) as f:
        config = json.load(f)
    # add prefix to directory path
    if 'prefix' in config:
        prefix = config['prefix']
        if not os.path.isabs(prefix):
            project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            prefix = os.path.join(project_dir, prefix)
        for dataset_name in config:
            if dataset_name == 'prefix':
                continue
            config[dataset_name]['directory'] = os.path.join(prefix, config[dataset_name]['directory'])
    return config
            
_DATASET_CONFIG = load_dataset_config()

def _get_dataset_prompts(dataset_func, dataset_config, use_generator, *args, **kwargs):
    return dataset_func(dataset_config, use_generator, *args, **kwargs)

def get_prompts_from_name(dataset_name:str, use_generator:bool=True, *args, **kwargs):  
    dataset_name_split = dataset_name.split('.')
    dataset_name = dataset_name_split[0]
    if len(dataset_name_split) > 1:
        kwargs["subtask_name"] = dataset_name_split[-1]
    return _get_dataset_prompts(NAME2DSFN[dataset_name], _DATASET_CONFIG[dataset_name], use_generator, *args, **kwargs)

def get_result_file_name(result_dir, world_size, use_assistant_model, model_name, time_str):
    pass

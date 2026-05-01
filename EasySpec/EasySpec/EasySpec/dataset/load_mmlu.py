from typing import Dict
from .dataset_utils import (
    load_from_local_hf,
    prompts_generator,
    _raw_to_chat_prompt_modifier
)
from .load_llama3eval import (
    llama3_eval_raw_to_chat_prompt_modifier,
    llama3_eval_message_decoder
)

# use cot at every inference
MMLU_INST = """
- For simple problems:
Directly provide the answer with minimal explanation.

- For complex problems:"""

def mmlu_raw_to_chat_prompt_modifier(raw_prompt_list, **prompt_modifier_kwargs):
    prompt = llama3_eval_raw_to_chat_prompt_modifier(raw_prompt_list, **prompt_modifier_kwargs)
    prompt = prompt.replace(MMLU_INST, "")
    return prompt

def get_mmlu_prompts(dataset_config: Dict, use_generator:bool, **kwargs):
    data = load_from_local_hf(dataset_config)
    prompt_modifier_kwargs = {
        "tokenizer": kwargs.pop("tokenizer"),
        "tokenizer_kwargs": kwargs.pop("tokenizer_kwargs"),
        "message_decoder": llama3_eval_message_decoder,
        "message_decoder_kwargs": {},
        # "eot_token_id": kwargs.pop("eot_token_id")
    }
    subtask_name = kwargs.pop("subtask_name", None)
    if subtask_name is None:
        data_final_prompts = data['input_final_prompts']
    else:
        data_final_prompts = (data_item['input_final_prompts'] for data_item in data if data_item['subtask_name'].split('.')[-1] == subtask_name)
    return prompts_generator(data_final_prompts, 
                             use_generator, 
                             mmlu_raw_to_chat_prompt_modifier, 
                             prompt_modifier_kwargs)
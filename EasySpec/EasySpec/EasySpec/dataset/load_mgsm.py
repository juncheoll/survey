from typing import Dict
from .dataset_utils import (
    load_from_local_hf,
    prompts_generator,
    _raw_to_chat_prompt_modifier
)
from .load_llama3eval import (
    llama3_eval_raw_to_chat_prompt_modifier,
    llama3_eval_message_decoder,
)

def mgsm_raw_to_chat_prompt_modifier(raw_prompt_list, **prompt_modifier_kwargs):
    prompt = llama3_eval_raw_to_chat_prompt_modifier(raw_prompt_list, **prompt_modifier_kwargs)
    prompt += '\nI am going to answer this question in your language, not in English.\n'
    return prompt

def get_mgsm_prompts(dataset_config: Dict, use_generator:bool, **kwargs):
    data = load_from_local_hf(dataset_config)
    prompt_modifier_kwargs = {
        "tokenizer": kwargs.pop("tokenizer"),
        "tokenizer_kwargs": kwargs.pop("tokenizer_kwargs"),
        "message_decoder": llama3_eval_message_decoder,
        "message_decoder_kwargs": {},
        # "eot_token_id": kwargs.pop("eot_token_id")
    }
    return prompts_generator(data['input_final_prompts'], 
                             use_generator, 
                             mgsm_raw_to_chat_prompt_modifier, 
                             prompt_modifier_kwargs)
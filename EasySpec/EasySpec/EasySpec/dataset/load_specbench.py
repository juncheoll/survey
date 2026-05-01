from typing import Dict
from .dataset_utils import (
    load_from_local_json,
    prompts_generator,
    _raw_to_chat_prompt_modifier,
    add_assistant_prompt_manual,
)


def load_from_local_specbench(dataset_config: Dict):
    data = load_from_local_json(dataset_config)
    return (data_item['turns'] for data_item in data if data_item['category'] == dataset_config['category'])

def specbench_message_decoder(prompt:str):
    # add roles to the messages
    messages = [
        {'role': 'user', 'content': prompt},
    ]
        
    return messages

def specbench_raw_to_chat_prompt_modifier(raw_prompt_list, **prompt_modifier_kwargs):
    if len(raw_prompt_list) > 1:
        raw_prompt_list = [raw_prompt_list[0]]
        # raise ValueError(f"specbench only supports len=1 prompt list, but {len(raw_prompt_list)}")
    if "tokenizer" not in prompt_modifier_kwargs:
        raise ValueError(f"must pass tokenizer to this modifier")
    
    raw_prompt = raw_prompt_list[0]
    prompt: str = _raw_to_chat_prompt_modifier(raw_prompt, **prompt_modifier_kwargs)
    prompt = add_assistant_prompt_manual(prompt)
    return prompt

def get_specbench_prompts(dataset_config: Dict, use_generator:bool, **kwargs):
    data = load_from_local_specbench(dataset_config)
    prompt_modifier_kwargs = {
        "tokenizer": kwargs.pop("tokenizer"),
        "tokenizer_kwargs": kwargs.pop("tokenizer_kwargs"),
        "message_decoder": specbench_message_decoder,
        "message_decoder_kwargs": {},
    }
    return prompts_generator(data, 
                             use_generator, 
                             specbench_raw_to_chat_prompt_modifier, 
                             prompt_modifier_kwargs)
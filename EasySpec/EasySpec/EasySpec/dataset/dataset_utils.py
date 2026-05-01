from datasets import load_dataset
from typing import Dict, List, Optional, Union
import os
import json

def load_from_local_hf(dataset_config: Dict):
    data = load_dataset(
        dataset_config['directory'], 
        name=dataset_config['name'], 
        split=dataset_config['split']
    )
    return data

def load_from_local_json(dataset_config: Dict):
    data_path = os.path.join(dataset_config['directory'], dataset_config['name'])
    data = (json.loads(x) for x in open(data_path).readlines())
    return data

def load_from_local_json_full(dataset_config: Dict):
    data_path = os.path.join(dataset_config['directory'], dataset_config['name'])
    with open(data_path, "r") as f:
        data = json.load(f)
    return data


def _no_modification_prompt_modifier(input_prompt, **kwargs):
    return input_prompt

def _no_modification_prompt_decoder(input_prompt, **kwargs):
    return input_prompt

def _raw_to_chat_prompt_modifier(
    raw_prompt: str,
    tokenizer, 
    tokenizer_kwargs, 
    message_decoder, 
    message_decoder_kwargs,
    **kwargs
):
    # generate a message
    """
        input_prompts is a string
        message_decoder is for raw prompt -> message
        tokenizer is for message -> chat prompt
    """ 
    res = tokenizer.apply_chat_template(message_decoder(raw_prompt, **message_decoder_kwargs), **tokenizer_kwargs)
    # TODO: more sophisticated way
    return res

def add_assistant_prompt_manual(prompt: str):
    if prompt.startswith("<|im_start|>"):
        # qwen
        prompt += "<|im_start|>assistant\n"
    elif prompt.startswith("<|begin_of_text|>"):
        # llama3
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        raise ValueError
    return prompt
        
def prompts_generator(
    input_prompts: List[str], 
    use_generator:bool,
    prompt_modifier, 
    prompt_modifier_kwargs, 
    ):
    if use_generator:
        def _generator():
            for input_prompt in input_prompts:
                yield prompt_modifier(input_prompt, **prompt_modifier_kwargs)
        return _generator()
    else:
        return [prompt_modifier(input_prompt, **prompt_modifier_kwargs) for input_prompt in input_prompts]


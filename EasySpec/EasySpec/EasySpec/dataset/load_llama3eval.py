from typing import Dict
from .dataset_utils import (
    load_from_local_hf,
    prompts_generator,
    _raw_to_chat_prompt_modifier
)

def llama3_eval_message_decoder(prompt:str):
    start_header_id = "<|start_header_id|>"
    end_header_id = "<|end_header_id|>"
    eot_id = "<|eot_id|>"
    
    prompt_split = prompt.split(eot_id)
    if len(prompt_split) <= 1:
        raise ValueError(prompt)
    
    # add roles to the messages
    messages = []
    for each_split in prompt_split:
        role, content = each_split.split(end_header_id)
        if not role.startswith(start_header_id):
            raise ValueError(role)
        role = role.replace("<|start_header_id|>", "")
        
        this_message = {"role":role, "content":content}
        messages.append(this_message)
    
    if messages[-2]['role'] != 'user' or messages[-1]['role'] != 'assistant':
        raise ValueError
    
    return messages

def llama3_eval_raw_to_chat_prompt_modifier(raw_prompt_list, **prompt_modifier_kwargs):
    if len(raw_prompt_list) > 1:
        raise ValueError(f"llama3 eval only supports len=1 prompt list, but {len(raw_prompt_list)}")
    if "tokenizer" not in prompt_modifier_kwargs:
        raise ValueError(f"must pass tokenizer to this modifier")
    tokenizer = prompt_modifier_kwargs["tokenizer"]
    raw_prompt = raw_prompt_list[0]
    raw_prompt: str = _raw_to_chat_prompt_modifier(raw_prompt, **prompt_modifier_kwargs)
    # prompt = raw_prompt
    
    # get appended string from chat_template
    _word = "Hello1121"
    _message = [{"role": "user", "content": _word}]
    eot_token_id_str = tokenizer.apply_chat_template(_message, tokenize=False).split(_word)[-1]
    
    if not raw_prompt.endswith(eot_token_id_str):
        # print("check tokenizer")
        prompt = raw_prompt
        # raise ValueError("check tokenizer, it does not output the correct eot token")
    else:
        prompt = raw_prompt[:-len(eot_token_id_str)]
    # print(prompt)
    return prompt

def get_llama3_eval_prompts(dataset_config: Dict, use_generator:bool, **kwargs):
    data = load_from_local_hf(dataset_config)
    prompt_modifier_kwargs = {
        "tokenizer": kwargs.pop("tokenizer"),
        "tokenizer_kwargs": kwargs.pop("tokenizer_kwargs"),
        "message_decoder": llama3_eval_message_decoder,
        "message_decoder_kwargs": {},
        
    }
    subtask_name = kwargs.pop("subtask_name", None)
    if subtask_name is None:
        data_final_prompts = data['input_final_prompts']
    else:
        data_final_prompts = (data_item['input_final_prompts'] for data_item in data if data_item['subtask_name'].split('.')[-1] == subtask_name)
        
    return prompts_generator(data_final_prompts, 
                             use_generator, 
                             llama3_eval_raw_to_chat_prompt_modifier, 
                             prompt_modifier_kwargs)
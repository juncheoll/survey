import torch
from typing import Dict, Optional
from ..generation.utils.base_parallel import BaseParallelGenerationMixin
from ..generation.utils.configuration_utils import GenerationConfig
from .base_parallel import BaseParallelPreTrainedModel
from ..generation.utils.tree_attention import (
    _update_tree_causal_mask_from_retrieve_indices,
)
from ..generation.utils.tree_attention import (
    _prepare_tree_verification_causal_mask
)
from ..generation.utils.attention_mask import (
    _prepare_4d_causal_attention_mask_with_cache_position    
)
    
def _prepare_position_id_from_2d_attention_mask(
    attention_mask: torch.Tensor,
) -> torch.LongTensor:
    """
        Return the position_ids inferred from 2d attention mask.
    """
    if attention_mask.dim() != 2:
        raise ValueError(f"preparing position id from attention mask, but attention mask dim is {attention_mask.dim()}")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids
    

_LLAMA_ATTN_MASK_DIM = 4
class LlamaGenerationMixin(BaseParallelGenerationMixin):
    """
        Mix:
            tensor parallel,
            hyperdraft
    """
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values,
        attention_mask: torch.Tensor,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        enable_tree_attention=False,
        **kwargs,
    ):
        if past_key_values is None:
            raise ValueError(f"Llama need to use cache.")
        if attention_mask is None:
            raise ValueError(f"Llama needs an attention mask, as it is a causal model.")
        elif attention_mask.dim() != 2 and attention_mask.dim() != _LLAMA_ATTN_MASK_DIM:
            raise ValueError(f"Llama needs a 2d or {_LLAMA_ATTN_MASK_DIM}d attention mask, but {attention_mask.dim()}.")
        if inputs_embeds is not None:
            raise NotImplementedError(f"No support for input_embeds for now.")

        # choose input_ids for the generation
        if input_ids.shape[1] != cache_position.shape[0]:
            # target_length = input_ids.shape[-1]
            input_ids = input_ids[:, cache_position]

        if position_ids is None:
            position_ids = _prepare_position_id_from_2d_attention_mask(attention_mask)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        if enable_tree_attention:
            # tree attention need 4d attention mask
            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=input_ids.shape[-1],
                target_length=attention_mask.shape[-1],
                dtype=self.info.dtype,
                device=self.info.device,
                cache_position=cache_position,
                batch_size=input_ids.shape[0],
            )
        
        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    def _update_model_kwargs_for_generation(
        self,
        outputs,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        num_new_tokens: int = 1,
    ):
        if getattr(outputs, "state", None) is not None:
            raise ValueError
        if "token_type_ids" in model_kwargs:
            raise ValueError
        
        # update past_key_values keeping its naming used in model code
        cache_name, cache = "past_key_values", outputs.past_key_values
        model_kwargs[cache_name] = cache
        
        enable_tree_attention = model_kwargs.get("enable_tree_attention")
        last_kv_len = model_kwargs["attention_mask"].shape[-1] # it is valid in both 2d and 4d case
        if not enable_tree_attention:
            # update attention mask
            attention_mask: torch.Tensor = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], num_new_tokens))], dim=-1
            )
            # in tree sampling, num_new_tokens is
            model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
        else:
            if generation_config.is_assistant:
                attention_mask = model_kwargs["attention_mask"]
                # it is a tree-sampling draft model
                # get retrieve_indices from model_kwargs
                topk_retrieve_indices: torch.LongTensor = model_kwargs["topk_retrieve_indices"]
                top_k = topk_retrieve_indices.shape[1]
                
                before_this_time_flat_tree_length = top_k * (topk_retrieve_indices.shape[2] - 1)
                non_tree_length = attention_mask.shape[-1] - before_this_time_flat_tree_length # it should be a constant
                
                # update attention mask
                model_kwargs["attention_mask"] = _update_tree_causal_mask_from_retrieve_indices(
                    attention_mask,
                    topk_retrieve_indices=topk_retrieve_indices,
                    non_tree_length=non_tree_length,
                    dtype=self.info.dtype,
                    device=self.info.device,
                )
                # update 2d attention mask
                attention_mask_2d: torch.LongTensor = model_kwargs["attention_mask_2d"]
                model_kwargs["attention_mask_2d"] = torch.cat(
                    [attention_mask_2d, attention_mask_2d.new_ones((attention_mask_2d.shape[0], num_new_tokens))], dim=-1
                )
                model_kwargs["cache_position"] = torch.arange(last_kv_len, last_kv_len + top_k, device=model_kwargs["cache_position"].device, dtype=model_kwargs["cache_position"].dtype)
                model_kwargs["position_ids"] = (model_kwargs["position_ids"][:,-1:] + 1).expand(model_kwargs["position_ids"].shape[0], top_k)
            else:
                # tree verification
                attention_mask_2d: torch.LongTensor = model_kwargs["attention_mask_2d"]
                # update attention mask
                model_kwargs["attention_mask_2d"] = torch.cat(
                    [attention_mask_2d, attention_mask_2d.new_ones((attention_mask_2d.shape[0], num_new_tokens))], dim=-1
                )
                model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + num_new_tokens
            
        return model_kwargs
    
    def _prepare_tree_verification_attention_mask(
        self,
        attention_mask_2d: torch.LongTensor,
        candidate_input_ids_length: int,
        flat_candidate_length: int,
        cache_position: torch.LongTensor,
        retrieve_indices: torch.Tensor,
        past_kv_length:int, 
        dtype: torch.dtype,
        device: torch.device,
    ):
        return _prepare_tree_verification_causal_mask(
            attention_mask_2d,
            candidate_input_ids_length=candidate_input_ids_length, 
            flat_candidate_length=flat_candidate_length,
            cache_position=cache_position,
            retrieve_indices=retrieve_indices, 
            past_kv_length=past_kv_length,
            dtype=dtype,
            device=device,
        )

class BaseParallelLlamaPretrainedModel(BaseParallelPreTrainedModel, LlamaGenerationMixin):
    # override some functions with LlamaGenerationMixin
    _model_type = 'llama'
    _support_tree_attention = True
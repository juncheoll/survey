import torch
from torch import nn
import torch.nn.functional as F
import math

import importlib
import importlib.metadata
from packaging import version

from ....utils import record_time_sync, rank0_print

from typing import Optional

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def compute_attn_output_torch_2_5(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, is_causal: bool):
    # start = record_time_sync()
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        is_causal=is_causal,
        enable_gqa=True
    )
    # rank0_print(f"mask dtype: {attention_mask.dtype}, attn time: {record_time_sync() - start}")
    return attn_output

def compute_attn_output(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, is_causal: bool):
    n_rep = q.shape[-3] // k.shape[-3]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        is_causal=is_causal,
    )
    return attn_output

def compute_attn_output_naive(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: torch.Tensor, is_causal: bool):
    
    n_rep = q.shape[-3] // k.shape[-3]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)
    
    attn_weight = (q @ k.transpose(-1,-2)) / math.sqrt(q.shape[-1])
    attn_output = (attn_weight + attention_mask).softmax(-1, dtype=torch.float32).to(q.dtype) @ v
    return attn_output

def is_version_greater_or_equal(library_name: str, library_version: str):
    return version.parse(importlib.metadata.version(library_name)) >= version.parse(library_version)

class SdpaAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # torch >= 2.5.0 supports group-query-attention
        # if is_version_greater_or_equal("torch", "2.5.0"):
        #     self.method = compute_attn_output_torch_2_5
        # else:
        #     self.method = compute_attn_output
        # self.method = compute_attn_output_naive
        self.method = compute_attn_output
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ):
        attn_output = self.method(
            q,
            k,
            v,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )
        return attn_output
    
def flatten_attn_output(attn_output: torch.Tensor):
    """
        [bsz, head_num, q_len, head_dim]
    """
    bsz, _, q_len, _ = attn_output.shape
    attn_output = attn_output.transpose(-2,-3).contiguous()
    return attn_output.reshape(bsz, q_len, -1)

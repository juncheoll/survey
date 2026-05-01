from typing import Any, Dict, List, Optional, Tuple
from ..utils import rank0_print
import torch
import torch.distributed as dist

from typing import List, Tuple, Dict, Optional, Any

def _tree_crop(
    hidden_states: torch.Tensor, 
    non_tree_length: int, 
    retrieve_indices:torch.Tensor
) -> torch.Tensor:
    
    to_select_tree_length = hidden_states.shape[-2] - non_tree_length
    # if no tree kv is cached, continue
    if to_select_tree_length <= 0:
        return hidden_states
    # retrieve tree part and cat to non-tree part
    non_tree_states, to_select_tree_states = hidden_states.split((non_tree_length, to_select_tree_length), dim=-2)
    if retrieve_indices.shape[-1] == 0:
        # discard all trees
        return non_tree_states
    # to_select_tree_states: [bsz, num_kv_head, to_select_len, head_dim]
    to_select_tree_states = to_select_tree_states[...,retrieve_indices[0],:]
    # to_select_tree_states: [bsz, num_kv_head, real_len, head_dim]
    hidden_states = torch.cat([non_tree_states, to_select_tree_states], dim=-2)
    return hidden_states


class TreeDynamicCache(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.seq_length = 0
        
    def __getitem__(self, layer_idx: int) -> List[Tuple[torch.Tensor]]:
        if layer_idx in self.key_cache:
            return (self.key_cache[layer_idx], self.value_cache[layer_idx])
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for layer_idx in self.key_cache:
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __len__(self):
        return len(self.key_cache)
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            What is different from Dynamic Cache?
            The seq_length will be updated.
        """        
        # Update the cache
        if layer_idx not in self.key_cache:
            if len(self.key_cache) == 0:
                self.seq_length += key_states.shape[-2]
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            maybe_new_seq_length = self.key_cache[layer_idx].shape[-2] + key_states.shape[-2]
            if maybe_new_seq_length > self.seq_length:
                self.seq_length = maybe_new_seq_length
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_length
    
    def get_min_length(self) -> int:
        min_length = -1
        for layer_idx in self.key_cache:
            if min_length == -1 or min_length > self.key_cache[layer_idx].shape[-2]:
                min_length = self.key_cache[layer_idx].shape[-2]
        if min_length == -1:
            raise ValueError
        return min_length

    def crop(self, max_length: int, valid_retrieve_indices: Optional[torch.Tensor] = None):
        """
            :
                if valid_retrieve_indices is None, crop invalid items
                if valid_retrieve_indices is not None, crop tree items
        """
        # In case it is negative
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        
        # Even if in tree crop, this is a valid return    
        if self.get_seq_length() <= max_length:
            return
        
        self.seq_length = max_length
        if valid_retrieve_indices is None:
            # caution: avoid referencing non-existing idx
            for idx in self.key_cache:
                self.key_cache[idx] = self.key_cache[idx][..., :max_length, :]
                self.value_cache[idx] = self.value_cache[idx][..., :max_length, :]
        else:
            # it is a tree-reserve crop
            non_tree_length = max_length - valid_retrieve_indices.shape[-1] 
            for idx in self.key_cache:
                self.key_cache[idx] = _tree_crop(self.key_cache[idx], non_tree_length, valid_retrieve_indices)
                self.value_cache[idx] = _tree_crop(self.value_cache[idx], non_tree_length, valid_retrieve_indices)
import torch
from torch import nn
from torch.nn.parameter import Parameter
from typing import List, Optional, Tuple, Union, Dict
import torch.nn.functional as F

from ..tp_module import TPModule
from ..weight_load_utils import create_param_tensor_on_device
from ...parallel.parallel_state import GroupCommunicator

class ColumnParallelLinear(TPModule):
    """
        (n*m, l) -> n*(m, l).
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = False, 
                 tp_group: Optional[GroupCommunicator] = None,
                 default_gather: bool = False):
        if tp_group is None:
            raise ValueError
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': None, 'dtype': None}
        
        self.tp_group = tp_group
        self.tp_size = self.tp_group.group_size
        self.default_gather = default_gather
        
        self.tp_out_features = out_features // self.tp_size
        if self.tp_out_features * self.tp_size != out_features:
            raise ValueError
        
        self.weight = Parameter(torch.empty(self.tp_out_features, in_features, **factory_kwargs))
        if bias == True:
            self.bias = Parameter(torch.empty((self.tp_out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        force_gather: bool = False
    ) -> torch.Tensor:
        # bias is of tp_out_features size, so just forward is ok
        hidden_states = F.linear(hidden_states, self.weight, self.bias)  
        if force_gather or self.default_gather:
            raise NotImplementedError
            # output_shape = list(hidden_states.shape)
            # output_shape[-1] *= self.tp_size
            # new_hidden_states = torch.empty(output_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        return hidden_states
    
    def weight_loader(self, module_name:str, value:torch.Tensor, dtype:torch.dtype, **kwargs):
        """
            value should be a safetensor slice.
        """
        old_value: torch.Tensor = getattr(self, module_name)
        # if module_name == 'bias':
        #     print(1)
        # load weight to device
        slice_start = self.tp_group.local_rank * self.tp_out_features
        slice_end = slice_start + self.tp_out_features
        # load tensor on
        value = value[slice_start:slice_end, ...]
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)
   
   
class RowParallelLinear(TPModule):
    """
        (m, n*l) -> n*(m, l).
    """
    _support_safetensor_slice = True
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = False, 
                 tp_group: Optional[GroupCommunicator] = None,
                 default_reduce: bool = True):
        if tp_group is None:
            raise ValueError
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': None, 'dtype': None}
        
        self.tp_group = tp_group
        self.tp_size = self.tp_group.group_size
        self.default_reduce = default_reduce
        
        self.tp_in_features = in_features // self.tp_size
        if self.tp_in_features * self.tp_size != in_features:
            raise ValueError
        
        self.weight = Parameter(torch.empty(self.out_features, self.tp_in_features, **factory_kwargs))
        if bias == True:
            self.bias = Parameter(torch.empty((self.out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        force_reduce: bool = False
    ) -> torch.Tensor:
        # matrix multiplication is tp, and bias is not
        hidden_states = F.linear(hidden_states, self.weight)  
        if self.default_reduce or force_reduce:
            self.tp_group.all_reduce(hidden_states)
        if self.bias is not None:
            hidden_states += self.bias
        return hidden_states
    
    def weight_loader(self, module_name:str, value, dtype:torch.dtype, **kwargs):
        """
            value should be a safetensor slice.
        """
        old_value: torch.Tensor = getattr(self, module_name)
        if module_name == 'bias':
            raise NotImplementedError
        # load weight to device
        
        slice_start = self.tp_group.local_rank * self.tp_in_features
        slice_end = slice_start + self.tp_in_features
        # load tensor on
        value = value[:, slice_start:slice_end]
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)
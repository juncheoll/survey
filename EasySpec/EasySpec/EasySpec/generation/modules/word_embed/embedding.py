import torch
from torch import nn
from ..weight_load_utils import create_param_tensor_on_device

class Embedding(nn.Embedding):
    def weight_loader(self, module_name, value, dtype, **kwargs):
        old_value = getattr(self, module_name)
        new_value = create_param_tensor_on_device(old_value, value, dtype)
        setattr(self, module_name, new_value)
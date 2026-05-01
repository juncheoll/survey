from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ReduceOp as TorchReduceOp

from enum import Enum

from .communicator import (
    GroupCommunicator,
    Layer2LayerGroupCommunicator,
    ReduceOp,
)
from .info import BaseInfo
from .parallel_utils import (
    get_global_rank,
    get_world_size,
)

_INFO: List[BaseInfo] = []   

def get_info(idx) -> BaseInfo:
    return _INFO[idx]

def register_info(info: BaseInfo):
    idx = len(_INFO)
    _INFO.append(info)
    return idx
            

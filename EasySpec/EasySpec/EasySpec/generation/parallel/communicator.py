import torch
import torch.distributed as dist
from torch.distributed import ReduceOp as TorchReduceOp
from .parallel_utils import get_global_rank, get_local_rank

from enum import Enum

from typing import Optional, List

class ReduceOp(Enum):
    SUM = 0
    AVG = 1

class GroupCommunicator:
    _comm_implementation = "torch"
    def __init__(self, group_ranks, driver=None):
        if self._comm_implementation != "torch":
            raise NotImplementedError
        if not isinstance(group_ranks, list):
            raise ValueError
        
        self.group_size = len(group_ranks)
        self.all_ranks = sorted(group_ranks) # guaranteed to be ordered
        self.rank = get_global_rank()
        self.in_participant = self.rank in self.all_ranks
        
        if driver is not None:
            if driver not in self.all_ranks:
                raise ValueError(f"all ranks are {self.all_ranks}, while driver is specific {driver} and not in.")
        else:
            # smallest rank as driver
            driver = self.all_ranks[0]
        self.driver = driver
        
        self.group = dist.new_group(self.all_ranks, use_local_synchronization=True)
        if self.group is None:
            if self.in_participant is True:
                # 1-process group
                print("You create a group communicator, but size is 1")
                raise ValueError("You create a group communicator, but size is 1")
                self.local_rank = 0
            else:
                self.local_rank = self.group_size # to raise exception if used
        else:
            # try one comm, to test if it is setup, and to avoid first-time latency of nccl
            self.test_group()
            self.local_rank = get_local_rank(self.group)
    
    def test_group(self):
        if self.group is None:
            raise ValueError
        obj = None
        obj_list = [obj]
        self.broadcast_object_list(obj_list, src=self.driver)
    
    def ignore_comm_op_or_raise(self):
        if self.group is None:
            if self.in_participant is True:
                # it is a one-process group
                return True
            else:
                raise ValueError()
        else:
            return False
        
    def all_reduce(self, t: torch.Tensor, op: Optional[ReduceOp] = None, async_op: bool = False):
        if self.ignore_comm_op_or_raise():
            return
        if op == ReduceOp.AVG:
            _op = TorchReduceOp.AVG
        elif op == ReduceOp.SUM:
            _op = TorchReduceOp.AVG
        else:
            _op = TorchReduceOp.SUM
        dist.all_reduce(t, _op, self.group, async_op=async_op)
    
    def broadcast(self, t: torch.Tensor, src:int, async_op: bool = False):
        if self.ignore_comm_op_or_raise():
            return
        dist.broadcast(t, src=src, group=self.group, async_op=async_op)
    
    def broadcast_object_list(self, obj_list, src:int, async_op:bool=False):
        if self.ignore_comm_op_or_raise():
            return
        if not isinstance(obj_list, list):
            raise ValueError
        dist.broadcast_object_list(obj_list, src=src, group=self.group)
    
    def all_gather_into_tensor(self, tensor_out:torch.Tensor, tensor_in:torch.Tensor, async_op:bool=False):
        if self.ignore_comm_op_or_raise():
            print("It is a one-process group, you could avoid using gathering to reduce communication.")
            tensor_out.copy_(tensor_in, non_blocking=True)
        dist.all_gather_into_tensor(tensor_out, tensor_in, self.group, async_op=async_op)
    
    def all_gather(self, tensor_list: List[torch.Tensor], t:torch.Tensor, async_op:bool=False):
        if self.ignore_comm_op_or_raise():
            return
        dist.all_gather(tensor_list, t, group=self.group, async_op=async_op)
    
    def barrier(self, async_op:bool=False):
        dist.barrier(self.group, async_op=async_op)
    

class Layer2LayerGroupCommunicator(GroupCommunicator):
    def __init__(self, dist_group, driver):
        super().__init__(dist_group)
        # Layer2Layer driver should be specified
        self.driver = driver

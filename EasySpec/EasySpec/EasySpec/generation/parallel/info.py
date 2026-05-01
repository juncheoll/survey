import torch
from typing import Optional, List

from .communicator import GroupCommunicator
from .parallel_utils import get_global_rank

def ready_device_list() -> List[int]:
    device_num = torch.cuda.device_count()
    # find ids of all visible devices
    device_list = []
    for i in range(device_num):
        try:
            _ = torch.tensor([0], device=i)
            device_list.append(i)
        except Exception:
            print(f"device {i} not working.")
            continue
    return device_list

_DEVICE_LIST = None

class _BaseInfo:
    def __init__(self, all_ranks, driver=None, device=None, dtype=None):
        if driver is not None and driver not in all_ranks:
            raise ValueError(f"driver {driver} is not in all ranks {all_ranks}")
        self.all_ranks = sorted(all_ranks) # make sure it is ordered
        self.in_participant = get_global_rank() in self.all_ranks
        if device is not None:
            if _DEVICE_LIST is None:
                _DEVICE_LIST = ready_device_list() 
            if device not in _DEVICE_LIST:
                raise ValueError(f"device is specified as {device}, while active devices are {_DEVICE_LIST}")
        self.device = device if device is not None else torch.cuda.current_device()
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.logits_dtype = torch.float32

class BaseInfo(_BaseInfo):
    """
        It has a group communicator.
    """
    def __init__(self, all_ranks, driver=None, device=None,):
        super().__init__(all_ranks, driver=driver, device=device)
        self.all_ranks_group = GroupCommunicator(all_ranks, driver=driver)
        self.driver = self.all_ranks_group.driver
        
class LlamalikeInfo(BaseInfo):
    """
        It specifies whether to broadcast input_ids to others.
    """
    def __init__(self, all_ranks, driver=None, device=None):
        super().__init__(all_ranks, driver=driver, device=device)
        # tp info does not need driver
        self.need_broadcast_inputs: bool = False
        
class TPInfo(LlamalikeInfo):
    def __init__(self, all_ranks, driver=None, device=None):
        super().__init__(all_ranks, driver=driver, device=device)
        # tp info does not need driver
        self.tp_group: GroupCommunicator = self.all_ranks_group

class SpeculativeDecodingInfo(BaseInfo):
    """
        SD info needs a group communicator, so use BaseInfo instead of _BaseInfo.
    """
    def __init__(self, draft_info:BaseInfo, verification_info:BaseInfo, device=None):
        if draft_info.driver != verification_info.driver:
            raise ValueError(f"Draft and verification model must have the same driver, but {draft_info.driver} and {verification_info.driver}")
        all_ranks = sorted(list(set(draft_info.all_ranks) | set(verification_info.all_ranks)))
        # If verification model all_ranks is a subset of draft model's, then no need to broadcast candidate
        self.need_broadcast_candidate_from_draft = (set(verification_info.all_ranks) - set(draft_info.all_ranks) != set())
        # self.need_broadcast_candidate_from_draft = True # TODO: maybe no need to broadcast?
        super().__init__(all_ranks, driver=draft_info.driver, device=device)

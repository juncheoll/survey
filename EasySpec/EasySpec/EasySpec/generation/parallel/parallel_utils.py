import torch.distributed as dist

def get_global_rank():
    return dist.get_rank()

def get_world_size(group=None):
    return dist.get_world_size(group)

def get_local_rank(group):
    return dist.get_group_rank(group, get_global_rank())

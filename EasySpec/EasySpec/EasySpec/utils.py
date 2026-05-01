import torch
import time
import torch.distributed as dist

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
        Compute the last 1-D's cosine similarity.
    """
    return (a * b).sum(dim=-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def record_time_sync():
    torch.cuda.current_stream().synchronize()
    return time.time()

def rank0_print(obj):
    if dist.get_rank() == 0:
        print(obj)

def print_and_record(fd, s:str):
    if fd is not None:
        fd.write(s + '\n')
    print(s)

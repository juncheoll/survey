import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
import os
import time

from transformers import AutoTokenizer

from ..auto.modeling_auto_distributed import AutoDistributedModelForCausalLM
from ...generation.parallel.communicator import GroupCommunicator
from ...generation.parallel.parallel_utils import get_global_rank, get_local_rank, get_world_size
from ...utils import record_time_sync, rank0_print

def visible_device_list(world_size):
    device_num = torch.cuda.device_count()
    if world_size > device_num:
        raise ValueError(f"world_size must be less or equal than # of current visible devices, but {world_size} vs {device_num}.")
    # find ids of all visible devices
    device_list = []
    for i in range(device_num):
        try:
            _ = torch.tensor([0], device=i)
            device_list.append(i)
            if len(device_list) == world_size:
                break
        except Exception:
            print(f"device {i} not working.")
            continue
    return device_list

_WORKER_INFO = None
class WorkerInfo:
    def __init__(self, world_size):
        # force driver to be 0
        self.tensor_group = GroupCommunicator([_ for _ in range(world_size)], driver=0) # a group for all processes
        self.group = self.tensor_group
        # self.non_tensor_group = dist.new_group([_ for _ in range(world_size)], backend='nccl')
        # test = [None]
        # dist.broadcast_object_list(test, src=0, group=self.non_tensor_group)
        # dist.barrier(self.non_tensor_group)

def init_env(global_rank, world_size, device):
    """
        For distributed environment, must set some environ variables.
    """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '49520'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(global_rank)
    torch.cuda.set_device(device)
    
def init_dist(global_rank, world_size, device):
    """
        Set up torch.distributed environment.
    """
    init_env(global_rank, world_size, device)
    dist.init_process_group(
        backend="nccl", 
        rank=global_rank, 
        world_size=world_size)
    global _WORKER_INFO
    _WORKER_INFO = WorkerInfo(world_size=world_size)

def broadcast_objects(src=0, group=None, objects=None):
    if not isinstance(objects, list):
        raise ValueError
    _WORKER_INFO.group.broadcast_object_list(objects, src=src)
    # dist.broadcast_object_list(objects, src=src, group=_WORKER_INFO.non_tensor_group)
    return objects

def broadcast_generation_args(generation_args, generation_kwargs):
    global_rank = get_global_rank()
    # start = record_time_sync()
    # print(f"rank {global_rank} start time: {start}")
        
    generation_args, = broadcast_objects(0, objects=[generation_args])
    # start = record_time_sync()
    # print(f"rank {global_rank} args time: {start}")
    generation_kwargs, = broadcast_objects(0, objects=[generation_kwargs])
    # start = record_time_sync()
    # print(f"rank {global_rank} kwargs time: {start}")
    
    return generation_args, generation_kwargs

def worker_func(global_rank, world_size, worker_pipe, device):
    
    init_dist(global_rank, world_size, device)    
    # batch group?
    print(f"rank {global_rank} finished worker initialization.")
    
    # rank 0 should communicate with host
    if global_rank == 0:
        from_pretrained_args, from_pretrained_kwargs = worker_pipe.recv()
    else:
        from_pretrained_args = None
        from_pretrained_kwargs = None
    
    objects = [from_pretrained_args, from_pretrained_kwargs]
    _WORKER_INFO.group.broadcast_object_list(objects, src=0)
    from_pretrained_args, from_pretrained_kwargs = objects
    
    # if assistant model is used, pop the args and kwargs
    assistant_model_path = from_pretrained_kwargs.pop("assistant_model_path", None)
    load_assistant_model = assistant_model_path is not None
    if load_assistant_model:
        assistant_from_pretrained_args = from_pretrained_kwargs.pop("assistant_from_pretrained_args", None)
        assistant_from_pretrained_kwargs = from_pretrained_kwargs.pop("assistant_from_pretrained_kwargs", None)
        # If there is no specific assistant args, use base model args
        if assistant_from_pretrained_args is None:
            assistant_from_pretrained_args = list(from_pretrained_args)
            assistant_from_pretrained_args[0] = assistant_model_path
        if assistant_from_pretrained_kwargs is None:
            assistant_from_pretrained_kwargs = from_pretrained_kwargs
            assistant_from_pretrained_kwargs["use_hyperdraft"] = True
            
    # load model
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    model_dir = from_pretrained_args[0] # make sure
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        *from_pretrained_args, **from_pretrained_kwargs
    )
    if load_assistant_model:
        # assistant_tokenizer = AutoTokenizer.from_pretrained(assistant_model_path)
        assistant_model = AutoDistributedModelForCausalLM.from_pretrained(
            assistant_model_path, **assistant_from_pretrained_kwargs
        )
    else:
        # assistant_tokenizer = None
        assistant_model = None
        
    model.prepare_for_assistant(assistant_model)
    _WORKER_INFO.tensor_group.barrier()
    del from_pretrained_args, from_pretrained_kwargs
    
    if global_rank == 0:
        worker_pipe.send("finish loading model.")
    
    my_generation_time = 0.0
    num_generated_tokens = 0
    # generate
    while True:
        if global_rank == 0:
            generation_args, generation_kwargs = worker_pipe.recv()
        else:
            generation_args, generation_kwargs = None, None
        
        generation_args, generation_kwargs = broadcast_generation_args(generation_args, generation_kwargs)
        if generation_args is None and generation_kwargs is None:
            # time to exit
            break
        
        # add assistant model into kwargs if needed
        use_assistant_model = generation_kwargs.get("use_assistant_model", False)
        if use_assistant_model is True:
            if assistant_model is None:
                raise ValueError(f"Cannot use assistant model. Not loaded.")
            generation_kwargs.update({"assistant_model": assistant_model})
        
        start = time.time()
        # print(f"rank {global_rank} start generate, time: {start}")
        # tokenize input prompts
        prompts = generation_args[0]
        for i,prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to('cuda')
            input_length = input_ids.shape[-1]
            
            # bsz = 1
            # input_ids = input_ids.repeat(bsz, 1)
            
            generation_args = [input_ids] + list(generation_args[1:])
            # add tokenizer to kwargs
            generation_kwargs["tokenizer"] = tokenizer
            
            # generate    
            res = model.generate(*generation_args, **generation_kwargs)
            
            num_generated_tokens += res.shape[-1] - input_length
            
        # print(f"rank {global_rank} end generate.")
        
        end = time.time()
        my_generation_time += end - start
        
        # send result back to host
        if global_rank == 0:
            res_dict = {
                "output_ids": res,
                "run_time": model.run_time,
                "this_run_time": model.this_run_time,
                "this_num_accepted_tokens": model.this_num_accepted_tokens,
                "this_num_candidate_tokens": model.this_num_candidate_tokens,
                "this_assistant_runtime": model.this_assistant_runtime,
                "this_ttft": getattr(model, "this_ttft", 0.0),
                "this_acceptance_steps": getattr(model, "this_acceptance_steps", []),
                "num_accepted_tokens": model.num_accepted_tokens,
                "num_candidate_tokens": model.num_candidate_tokens,
                "assistant_runtime": model.assistant_runtime,
            }
            worker_pipe.send(res_dict)
    
    full_generation_time = torch.tensor(my_generation_time, dtype=torch.float32, device=dist.get_rank())
    # print(f"rank {global_rank} generation time: {my_generation_time}")
    assistant_model_run_time = model.assistant_runtime
    base_model_run_time = my_generation_time - model.assistant_runtime
    # print(f"rank {global_rank} assistant run time: {assistant_model_run_time}")
    # print(f"rank {global_rank} base model raw run time: {base_model_run_time}")
    dist.reduce(full_generation_time, dst=0, op=dist.ReduceOp.SUM)
    if global_rank == 0:
        worker_pipe.send([
            my_generation_time, 
            num_generated_tokens, 
            model.num_accepted_tokens, 
            model.num_candidate_tokens, 
            assistant_model_run_time
        ])
    
    del generation_args, generation_kwargs, model, res
    if global_rank == 0:
        worker_pipe.close()
    
    print(f"rank {global_rank} destroying process group")
    dist.destroy_process_group()

def init_workers(world_size: int):
    
    print("start initializing worker")
    device_list = visible_device_list(world_size)
    
    mp.set_start_method("spawn", force=True)
    
    host_pipe, worker_pipe = mp.Pipe()
    workers = []
    for global_rank in range(world_size):
        p = mp.Process(target=worker_func, args=(global_rank, world_size, worker_pipe, device_list[global_rank]))
        workers.append(p)
    
    for worker in workers:
        worker.start()
        
    # remember to put worker in model.workers
    return workers, host_pipe

def destroy_workers(workers, host_pipe):
    for worker in workers:
        worker.join()
    host_pipe.close()

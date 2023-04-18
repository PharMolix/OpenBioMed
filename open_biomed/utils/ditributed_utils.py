import os

import torch
import torch.distributed as dist

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def is_main_process():
    return get_rank() == 0

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if not args.distributed:
        args.device = 0
        return
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.device = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(args.device)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                             world_size=args.world_size, rank=args.rank)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def mean_reduce(val, cur_device, dest_device, world_size):
    val = val.clone().detach() if torch.is_tensor(val) else torch.tensor(val)
    val = val.to(cur_device)
    torch.distributed.reduce(val, dst=dest_device)
    return val.item() / world_size

def concat_reduce(tensor, num_total_examples, world_size):
    output_tensors = [tensor.clone() for _ in range(world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]
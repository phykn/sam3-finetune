import torch.distributed as dist


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if active() else 1


def sum_value(value):
    out = value.detach().clone()
    if active():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out

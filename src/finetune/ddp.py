import os

import torch
import torch.distributed as dist


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if active() else 1


def rank() -> int:
    return dist.get_rank() if active() else 0


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def init() -> torch.device:
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local = local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        backend = "nccl"
        device = torch.device("cuda", local)
    else:
        backend = "gloo"
        device = torch.device("cpu")
    dist.init_process_group(backend=backend, init_method="env://")
    return device


def sum_value(value):
    out = value.detach().clone()
    if active():
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def broadcast_object(value):
    if not active():
        return value
    values = [value if is_main() else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def all_finite(value: torch.Tensor) -> bool:
    flag = torch.tensor(
        int(torch.isfinite(value).all()),
        device=value.device,
    )
    if active():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def finish() -> None:
    if active():
        dist.destroy_process_group()

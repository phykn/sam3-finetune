from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.data.dataloader import InfiniteLoader
from src.finetune import ddp


class FakeSampler:
    def __init__(self) -> None:
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


def test_ddp_helpers_default_to_single_process():
    assert ddp.active() is False
    assert ddp.rank() == 0
    assert ddp.local_rank() == 0
    assert ddp.world_size() == 1
    assert ddp.is_main() is True
    assert ddp.broadcast_object("run") == "run"
    assert ddp.all_finite(torch.tensor(1.0)) is True


def test_infinite_loader_advances_distributed_sampler_epoch():
    sampler = FakeSampler()
    loader = InfiniteLoader([[1]], sampler=sampler)

    assert next(loader) == [1]
    assert next(loader) == [1]
    assert sampler.epochs == [1]


def _gloo_worker(rank, world_size, init_file, result_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file:///{Path(init_file).as_posix()}",
        rank=rank,
        world_size=world_size,
    )
    torch.manual_seed(rank + 1)
    model = nn.Linear(1, 1, bias=False)
    model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    value = torch.tensor([[float(rank + 1)]])

    optimizer.zero_grad(set_to_none=True)
    model(value).square().sum().backward()
    optimizer.step()
    torch.save(model.module.weight.detach(), Path(result_dir) / f"rank-{rank}.pt")
    if ddp.is_main():
        (Path(result_dir) / "rank-zero.txt").touch()
    ddp.finish()


def test_two_process_gloo_keeps_parameters_equal_and_one_main(tmp_path):
    init_file = tmp_path / "gloo-init"
    mp.spawn(
        _gloo_worker,
        args=(2, str(init_file), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    first = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    second = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert torch.equal(first, second)
    assert (tmp_path / "rank-zero.txt").is_file()

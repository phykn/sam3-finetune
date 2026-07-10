import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build import build_finetune_loader, build_finetune_model  # noqa: E402
from src.finetune import ddp  # noqa: E402
from src.finetune.checkpoint import load_checkpoint  # noqa: E402
from src.finetune.trainer import FinetuneTrainer  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as file:
        return yaml.safe_load(file)


def resume_run_dir(path: str | Path) -> Path:
    return Path(path).resolve().parent.parent


def new_run_dir(root: str | Path) -> Path:
    value = None
    if ddp.is_main():
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        value = str(Path(root).resolve() / timestamp)
    return Path(ddp.broadcast_object(value))


def run(
    config: dict,
    device: torch.device,
    resume: str | Path | None = None,
) -> dict[str, float]:
    model_config = dict(config["model"])
    model_config["device"] = device
    model = build_finetune_model(model_config)
    train_loader = build_finetune_loader(
        config["data"]["train"],
        train=True,
        rank=ddp.rank(),
        world_size=ddp.world_size(),
    )
    valid_loader = build_finetune_loader(
        config["data"]["valid"],
        train=False,
        rank=ddp.rank(),
        world_size=ddp.world_size(),
    )
    train_config = config["train"]
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=train_config["learning_rate"],
    )

    step = 0
    if resume is not None:
        step, _saved_config = load_checkpoint(resume, model, optimizer)
        run_dir = resume_run_dir(resume)
    else:
        run_dir = new_run_dir(train_config.get("run_root", "run"))

    if ddp.active():
        device_ids = [ddp.local_rank()] if device.type == "cuda" else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            broadcast_buffers=False,
        )

    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        steps=train_config["steps"],
        valid_steps=train_config["valid_steps"],
        device=device,
        run_dir=run_dir,
        save_every=train_config["save_every"],
        clip_grad_norm=train_config.get("clip_grad_norm"),
        amp=train_config.get("amp", True),
        step=step,
        config=config,
    )
    try:
        return trainer.train()
    finally:
        trainer.close()


def main():
    args = parse_args()
    device = ddp.init()
    try:
        run(load_config(args.config), device, resume=args.resume)
    finally:
        ddp.finish()


if __name__ == "__main__":
    main()

from itertools import cycle

import torch
from torch import nn

from src.finetune.trainer import FinetuneTrainer


class TinyFinetuneModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.scale = nn.Parameter(torch.tensor(0.0))
        self.object_bias = nn.Parameter(torch.tensor(0.0))
        self.class_head = nn.Linear(1, 2)

    def forward(self, batch):
        image = batch["image"]
        pooled = image.flatten(1).mean(dim=1, keepdim=True)
        return {
            "mask_logits": image[:, :1] * self.scale + self.object_bias,
            "iou_scores": pooled * self.scale + self.object_bias,
            "class_logits": (self.class_head(pooled) + self.scale).unsqueeze(1),
        }


def make_batch() -> dict:
    return {
        "image": torch.ones(2, 1, 4, 4),
        "target": torch.zeros(2, 1, 4, 4),
        "mask_valid": torch.tensor([1.0, 0.0]),
        "is_auto_bg": torch.tensor([0.0, 1.0]),
        "label_target": torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
        "label_weight": torch.tensor([[1.0, 1.0], [1.0, 0.0]]),
        "prompt": [{"type": "point"}, {"type": "box"}],
    }


def test_train_step_updates_trainable_params_and_saves_checkpoint(tmp_path):
    model = TinyFinetuneModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FinetuneTrainer(
        model=model,
        train_loader=cycle([make_batch()]),
        valid_loader=cycle([make_batch()]),
        optimizer=optimizer,
        steps=1,
        valid_steps=1,
        device="cpu",
        run_root=tmp_path,
    )
    before = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    frozen_before = model.frozen.detach().clone()

    stats = trainer.train_step()
    checkpoint = torch.load(
        trainer.checkpoint_dir / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    changed = any(
        not torch.allclose(before[name], param.detach())
        for name, param in model.named_parameters()
        if param.requires_grad
    )
    assert changed
    assert torch.equal(model.frozen.detach(), frozen_before)
    assert trainer.step == 1
    assert set(stats) == {
        "loss",
        "mask_bce",
        "mask_dice",
        "iou_loss",
        "class_loss",
        "grad_norm",
    }
    assert checkpoint["step"] == 1
    assert "optimizer" in checkpoint
    assert "frozen" not in checkpoint["model"]
    assert "scale" in checkpoint["model"]


def test_train_runs_fixed_steps_and_saves_interval_checkpoint(tmp_path):
    model = TinyFinetuneModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FinetuneTrainer(
        model=model,
        train_loader=[make_batch()],
        valid_loader=[make_batch()],
        optimizer=optimizer,
        steps=2,
        valid_steps=1,
        device="cpu",
        run_root=tmp_path,
        save_every=2,
    )

    stats = trainer.train()
    trainer.close()

    assert trainer.step == 2
    assert "loss" in stats
    assert "valid_loss" in stats
    assert (trainer.checkpoint_dir / "step-000002.pt").is_file()
    assert not (trainer.checkpoint_dir / "step-000001.pt").exists()
    assert list(trainer.log_dir.glob("events.out.tfevents.*"))


def test_validation_runs_on_checkpoint_interval(tmp_path):
    model = TinyFinetuneModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FinetuneTrainer(
        model=model,
        train_loader=cycle([make_batch()]),
        valid_loader=cycle([make_batch()]),
        optimizer=optimizer,
        steps=3,
        valid_steps=2,
        device="cpu",
        run_root=tmp_path,
        save_every=2,
    )

    stats = trainer.train()
    trainer.close()

    assert trainer.step == 3
    assert trainer.valid_count == 1
    assert stats["valid_loss"] > 0


def test_valid_step_does_not_update_model(tmp_path):
    model = TinyFinetuneModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FinetuneTrainer(
        model=model,
        train_loader=cycle([make_batch()]),
        valid_loader=cycle([make_batch()]),
        optimizer=optimizer,
        steps=1,
        valid_steps=1,
        device="cpu",
        run_root=tmp_path,
    )
    before = [param.detach().clone() for param in model.parameters()]

    stats = trainer.valid_step()
    trainer.close()

    after = [param.detach() for param in model.parameters()]
    assert all(torch.equal(old, new) for old, new in zip(before, after))
    assert set(stats) == {
        "loss",
        "mask_bce",
        "mask_dice",
        "iou_loss",
        "class_loss",
    }


def test_finetune_package_exports_trainer():
    from src.finetune import FinetuneTrainer as Exported

    assert Exported is FinetuneTrainer


def test_resume_step_runs_only_remaining_global_steps(tmp_path):
    model = TinyFinetuneModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = FinetuneTrainer(
        model=model,
        train_loader=cycle([make_batch()]),
        valid_loader=cycle([make_batch()]),
        optimizer=optimizer,
        steps=3,
        valid_steps=1,
        device="cpu",
        run_root=tmp_path,
        save_every=2,
        step=2,
        config={"train": {"steps": 3}},
    )

    trainer.train()
    trainer.close()

    checkpoint = torch.load(
        trainer.checkpoint_dir / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert trainer.step == 3
    assert checkpoint["step"] == 3
    assert checkpoint["config"] == {"train": {"steps": 3}}

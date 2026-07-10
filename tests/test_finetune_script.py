from pathlib import Path

import torch
import yaml
from torch import nn


def test_finetune_config_has_one_complete_tree():
    config = yaml.safe_load(Path("config/finetune.yaml").read_text(encoding="utf-8"))

    assert set(config) == {"model", "data", "train"}
    assert set(config["data"]) == {"train", "valid"}
    assert "path" in config["model"]
    assert "steps" in config["train"]


def test_resume_run_dir_is_checkpoint_run_dir(tmp_path):
    from scripts.finetune import resume_run_dir

    path = tmp_path / "run-a" / "checkpoints" / "last.pt"

    assert resume_run_dir(path) == tmp_path / "run-a"


def test_run_loads_resume_before_ddp_and_passes_step(monkeypatch, tmp_path):
    import scripts.finetune as script

    events = []
    captured = {}

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.value = nn.Parameter(torch.tensor(1.0))

        def trainable_parameters(self):
            return [self.value]

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def train(self):
            events.append("train")

        def close(self):
            events.append("close")

    def load_checkpoint(_path, _model, _optimizer):
        events.append("load")
        return 7, {"saved": True}

    def wrap(model, **_kwargs):
        events.append("wrap")
        return model

    monkeypatch.setattr(script, "build_finetune_model", lambda _config: FakeModel())
    monkeypatch.setattr(script, "build_finetune_loader", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(script, "load_checkpoint", load_checkpoint)
    monkeypatch.setattr(script, "DistributedDataParallel", wrap)
    monkeypatch.setattr(script, "FinetuneTrainer", FakeTrainer)
    monkeypatch.setattr(script.ddp, "active", lambda: True)
    monkeypatch.setattr(script.ddp, "rank", lambda: 0)
    monkeypatch.setattr(script.ddp, "world_size", lambda: 2)
    monkeypatch.setattr(script.ddp, "local_rank", lambda: 0)
    resume = tmp_path / "run-a" / "checkpoints" / "last.pt"
    config = {
        "model": {"path": "base.pt"},
        "data": {"train": {}, "valid": {}},
        "train": {
            "steps": 10,
            "valid_steps": 1,
            "learning_rate": 1e-4,
            "save_every": 2,
            "clip_grad_norm": 1.0,
            "amp": False,
            "run_root": str(tmp_path),
        },
    }

    script.run(config, torch.device("cpu"), resume=resume)

    assert events == ["load", "wrap", "train", "close"]
    assert captured["step"] == 7
    assert captured["run_dir"] == tmp_path / "run-a"
    assert captured["config"] == config

from collections.abc import Iterable
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from .loss import auto_bg_label_weight, label_loss, noisy_mask_loss


class FinetuneTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: Iterable,
        valid_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        steps: int,
        valid_steps: int,
        device: str | torch.device,
        run_root: str | Path = "run",
        run_dir: str | Path | None = None,
        save_every: int = 1,
        clip_grad_norm: float | None = 1.0,
        amp: bool = True,
    ) -> None:
        if steps <= 0:
            raise ValueError("steps must be positive.")
        if valid_steps <= 0:
            raise ValueError("valid_steps must be positive.")
        if save_every <= 0:
            raise ValueError("save_every must be positive.")
        if clip_grad_norm is not None and clip_grad_norm <= 0:
            raise ValueError("clip_grad_norm must be positive or None.")

        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_iterator = iter(train_loader)
        self.valid_iterator = iter(valid_loader)
        self.optimizer = optimizer
        self.steps = steps
        self.valid_steps = valid_steps
        self.save_every = save_every
        self.clip_grad_norm = clip_grad_norm
        self.device = torch.device(device)
        self.amp = amp
        self.step = 0
        self.valid_count = 0
        self.valid_stats: dict[str, float] = {}

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        if run_dir is None:
            self.run_dir = Path(run_root) / timestamp
        else:
            self.run_dir = Path(run_dir)
        self.log_dir = self.run_dir / "log" / "finetune"
        self.weight_dir = self.run_dir / "weight" / "finetune"
        self.last_weight_dir = self.weight_dir / "last"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_weight_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def train_step(self) -> dict[str, float]:
        self.model.train()
        batch, self.train_iterator = self._next_batch(
            self.train_loader,
            self.train_iterator,
        )
        batch = self._to_device(batch)

        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast():
            out = self.model(batch)
            loss, stats = self._loss(batch, out)

        loss.backward()
        grad_norm = self.grad_norm()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
        self.optimizer.step()

        self.step += 1
        self.save_checkpoint()
        stats["grad_norm"] = grad_norm
        self._log("train", stats)
        return stats

    def valid_step(self) -> dict[str, float]:
        self.model.eval()
        batch, self.valid_iterator = self._next_batch(
            self.valid_loader,
            self.valid_iterator,
        )
        batch = self._to_device(batch)

        with torch.no_grad(), self._autocast():
            out = self.model(batch)
            _, stats = self._loss(batch, out)

        return stats

    def validate(self) -> dict[str, float]:
        total: dict[str, float] = {}
        for _ in range(self.valid_steps):
            stats = self.valid_step()
            for key, value in stats.items():
                total[key] = total.get(key, 0.0) + value

        self.valid_stats = {
            key: value / self.valid_steps for key, value in total.items()
        }
        self.valid_count += 1
        self._log("valid", self.valid_stats)
        return self.valid_stats

    def train(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        progress = tqdm(range(self.steps), total=self.steps, desc="finetune")
        for _ in progress:
            stats = self.train_step()
            if self.step % self.save_every == 0:
                self.validate()

            stats = self._with_valid_stats(stats)
            progress.set_postfix(
                {key: f"{value:.4g}" for key, value in stats.items()}
            )
        return stats

    def save_checkpoint(self) -> None:
        checkpoint = {
            "step": self.step,
            "model": self._model_state(),
            "optimizer": self.optimizer.state_dict(),
        }
        if self.step % self.save_every == 0:
            self._save_checkpoint_file(
                checkpoint,
                self.weight_dir / str(self.step) / "model.pt",
            )
        self._save_checkpoint_file(checkpoint, self.last_weight_dir / "model.pt")

    def _loss(
        self,
        batch: dict[str, Any],
        out: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mask = noisy_mask_loss(
            out["mask_logit"].float(),
            batch["target"].float(),
            batch["has_mask"],
        )
        label_weight = auto_bg_label_weight(
            batch["label_weight"].float(),
            out["object_logit"].float(),
            batch["is_auto_bg"],
        )
        label = label_loss(
            out["label_logit"].float(),
            batch["label_target"].float(),
            label_weight,
        )
        loss = mask + label
        return loss, {
            "loss": float(loss.detach().cpu()),
            "mask_loss": float(mask.detach().cpu()),
            "label_loss": float(label.detach().cpu()),
        }

    def _with_valid_stats(self, stats: dict[str, float]) -> dict[str, float]:
        out = dict(stats)
        out.update({f"valid_{key}": value for key, value in self.valid_stats.items()})
        return out

    def _log(self, prefix: str, stats: dict[str, float]) -> None:
        for key, value in stats.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, self.step)

    def close(self) -> None:
        self.writer.close()

    def _next_batch(self, loader: Iterable, iterator: Any) -> tuple[Any, Any]:
        try:
            return next(iterator), iterator
        except StopIteration:
            iterator = iter(loader)
            return next(iterator), iterator

    def _to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._to_device(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._to_device(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._to_device(item) for item in value)
        return value

    def _model_state(self) -> dict[str, torch.Tensor]:
        model = self.model
        while hasattr(model, "module") and isinstance(model.module, nn.Module):
            model = model.module
        return {
            name: param.detach().cpu()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def _save_checkpoint_file(self, checkpoint: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)

    def grad_norm(self) -> float:
        total = 0.0
        for param in self.model.parameters():
            if param.grad is None:
                continue
            norm = param.grad.detach().data.norm(2).item()
            total += norm * norm
        return total**0.5

    def _autocast(self) -> Any:
        if self.amp and self.device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

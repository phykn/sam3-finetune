from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class GroundingPrediction:
    masks: np.ndarray
    mask_logits: np.ndarray
    boxes_xyxy: np.ndarray
    scores: np.ndarray


@dataclass(frozen=True)
class VisualLanguageCache:
    language_features: torch.Tensor
    language_mask: torch.Tensor
    language_embeds: torch.Tensor | None = None

    @classmethod
    def from_file(cls, path: str | Path) -> "VisualLanguageCache":
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if "language_features" not in cache or "language_mask" not in cache:
            raise ValueError(
                "visual cache must contain language_features and language_mask"
            )
        return cls(
            language_features=cache["language_features"],
            language_mask=cache["language_mask"],
            language_embeds=cache.get("language_embeds"),
        )

    def to_backbone_out(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        output = {
            "language_features": self.language_features.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            ),
            "language_mask": self.language_mask.to(
                device=device,
                dtype=torch.bool,
                non_blocking=True,
            ),
        }
        if self.language_embeds is not None:
            output["language_embeds"] = self.language_embeds.to(
                device=device,
                dtype=dtype,
                non_blocking=True,
            )
        return output

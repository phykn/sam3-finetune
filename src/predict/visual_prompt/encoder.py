from collections.abc import Sequence

import numpy as np
import torch

from .types import PreparedVisualConcept, PreparedVisualPrompts, VisualExemplar

VISUAL_PROMPT_TEXT = "visual"


class VisualPromptEncoder:
    def __init__(
        self,
        grounding_model: torch.nn.Module,
        *,
        device: torch.device | str = "cuda",
        image_size: int = 1008,
    ) -> None:
        self.model = grounding_model
        self.device = torch.device(device)
        self.image_size = int(image_size)

    @torch.inference_mode()
    def prepare(self, exemplars: Sequence[VisualExemplar]) -> PreparedVisualPrompts:
        if not exemplars:
            raise ValueError("exemplars must be non-empty")
        if not hasattr(self.model.backbone, "forward_text"):
            raise RuntimeError("visual prompts require a VLM language backbone")

        grouped: dict[int, list[VisualExemplar]] = {}
        for exemplar in exemplars:
            grouped.setdefault(int(exemplar.concept_id), []).append(exemplar)

        concepts: list[PreparedVisualConcept] = []
        for concept_id, group in grouped.items():
            boxes = torch.stack(
                [_mask_to_normalized_cxcywh(exemplar.mask) for exemplar in group],
                dim=0,
            ).view(len(group), 1, 4)
            labels = torch.ones(len(group), 1, dtype=torch.bool)
            language = self.model.backbone.forward_text(
                [VISUAL_PROMPT_TEXT],
                device=self.device,
            )
            concepts.append(
                PreparedVisualConcept(
                    concept_id=concept_id,
                    exemplars=tuple(group),
                    boxes_cxcywh=boxes.to(self.device),
                    box_labels=labels.to(self.device),
                    language_features=language["language_features"].to(self.device),
                    language_mask=language["language_mask"].to(self.device),
                    language_embeds=(
                        None
                        if "language_embeds" not in language
                        else language["language_embeds"].to(self.device)
                    ),
                )
            )
        return PreparedVisualPrompts(concepts=tuple(concepts))


def _mask_to_normalized_cxcywh(mask: np.ndarray) -> torch.Tensor:
    ys, xs = np.nonzero(mask)
    if xs.size == 0 or ys.size == 0:
        raise ValueError("mask must contain foreground pixels")
    height, width = mask.shape
    x0 = float(xs.min())
    y0 = float(ys.min())
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    return torch.tensor(
        [
            ((x0 + x1) * 0.5) / float(width),
            ((y0 + y1) * 0.5) / float(height),
            (x1 - x0) / float(width),
            (y1 - y0) / float(height),
        ],
        dtype=torch.float32,
    )

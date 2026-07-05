from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..prompted import Sam3Predictor


@dataclass(frozen=True)
class RefinedMaskResult:
    mask: np.ndarray
    score: float
    low_res_mask: np.ndarray
    selected_index: int


def select_best_mask(
    masks: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    mask_arr = np.asarray(masks)
    score_arr = np.asarray(scores, dtype=np.float32)
    if mask_arr.ndim < 3:
        raise ValueError("masks must have candidate and spatial dimensions")

    candidate_count = int(np.prod(mask_arr.shape[:-2]))
    flat_scores = score_arr.reshape(-1)
    if flat_scores.size != candidate_count:
        raise ValueError(
            f"score count {flat_scores.size} does not match mask count {candidate_count}"
        )

    flat_masks = mask_arr.reshape(candidate_count, *mask_arr.shape[-2:])
    selected_index = int(np.argmax(flat_scores))
    return (
        flat_masks[selected_index].astype(bool),
        float(flat_scores[selected_index]),
        selected_index,
    )


class MaskRefiner:
    def __init__(self, predictor) -> None:
        self.predictor = predictor

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "MaskRefiner":
        return cls(Sam3Predictor.from_checkpoint(path, device=device))

    def refine(
        self,
        *,
        image=None,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor,
    ) -> RefinedMaskResult:
        if mask_input is None:
            raise ValueError("mask_input is required for refinement")
        if image is not None:
            self.predictor.set_image(image)
        masks, scores, low_res_masks = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=False,
        )
        mask, score, selected_index = select_best_mask(masks, scores)
        flat_low_res = np.asarray(low_res_masks).reshape(
            -1,
            *np.asarray(low_res_masks).shape[-2:],
        )
        return RefinedMaskResult(
            mask=mask,
            score=score,
            low_res_mask=flat_low_res[selected_index],
            selected_index=selected_index,
        )

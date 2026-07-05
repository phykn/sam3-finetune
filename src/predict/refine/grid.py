from dataclasses import dataclass
from pathlib import Path

import torch

from ...types import ContextPrediction, ContextReference, MaskInstance
from ..context import ContextMatcher
from ..context.prototype import build_context_prototype, select_feature, similarity_map
from ..grid import AutomaticMaskGenerator
from ..prompted import Sam3Predictor


@dataclass(frozen=True)
class ContextGridRefineResult:
    base_instances: list[MaskInstance]
    context_references: list[ContextReference]
    refined_predictions: list[ContextPrediction]


class ContextGridRefiner:
    def __init__(
        self,
        *,
        base_generator,
        matcher,
    ) -> None:
        self.base_generator = base_generator
        self.matcher = matcher

    @classmethod
    def from_predictor(
        cls,
        predictor,
        *,
        grid_kwargs: dict | None = None,
        matcher_kwargs: dict | None = None,
    ) -> "ContextGridRefiner":
        base_generator = AutomaticMaskGenerator(
            predictor,
            **(grid_kwargs or {}),
        )
        matcher = ContextMatcher(
            predictor,
            **(matcher_kwargs or {}),
        )
        return cls(base_generator=base_generator, matcher=matcher)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        *,
        grid_kwargs: dict | None = None,
        matcher_kwargs: dict | None = None,
    ) -> "ContextGridRefiner":
        predictor = Sam3Predictor.from_checkpoint(path, device=device)
        return cls.from_predictor(
            predictor,
            grid_kwargs=grid_kwargs,
            matcher_kwargs=matcher_kwargs,
        )

    @torch.inference_mode()
    def refine(
        self,
        image,
        *,
        max_masks: int | None = None,
    ) -> ContextGridRefineResult:
        base_instances = self.base_generator.generate_instances(image)
        context_references = grid_instances_to_context_references(
            image,
            base_instances,
        )
        if not context_references:
            return ContextGridRefineResult(
                base_instances=base_instances,
                context_references=[],
                refined_predictions=[],
            )
        refined_predictions = self._predict_context(
            image,
            context_references,
            max_masks=max_masks,
        )
        return ContextGridRefineResult(
            base_instances=base_instances,
            context_references=context_references,
            refined_predictions=refined_predictions,
        )

    def _predict_context(
        self,
        image,
        context_references: list[ContextReference],
        *,
        max_masks: int | None,
    ) -> list[ContextPrediction]:
        if not isinstance(self.matcher, ContextMatcher):
            return self.matcher.predict(
                image,
                context_references,
                max_masks=max_masks,
            )

        embedding = self.matcher.predictor.encode_image(image)
        prototype = build_context_prototype(
            context_references,
            [embedding] * len(context_references),
            feature_layer=self.matcher.feature_layer,
            negative_context_mode=self.matcher.negative_context_mode,
            negative_context_scale=self.matcher.negative_context_scale,
        )
        features = select_feature(embedding, self.matcher.feature_layer)
        similarity = similarity_map(
            features,
            prototype,
            negative_context_weight=self.matcher.negative_context_weight,
        )
        candidate_score_map = self.matcher._candidate_score_map(
            similarity,
            shape_prior=None,
        )
        point_coords = self.matcher._candidate_points(
            candidate_score_map,
            embedding.orig_hw,
        )
        if len(point_coords) == 0:
            return []
        return self.matcher._decode_candidates(
            embedding,
            point_coords,
            similarity,
            reference_area_ratio=prototype.reference_area_ratio,
            shape_prior=None,
            max_masks=self.matcher.max_masks if max_masks is None else int(max_masks),
        )


def grid_instances_to_context_references(
    image,
    instances: list[MaskInstance],
) -> list[ContextReference]:
    return [
        ContextReference(
            image=image,
            mask=instance.to_full_mask(),
        )
        for instance in instances
    ]

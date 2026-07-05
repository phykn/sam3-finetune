from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...types import (
    ContextReference,
    MaskInstance,
    ReferenceExample,
    Sam3ImageEmbedding,
)
from ..prompted import Sam3Predictor
from .prototype import (
    build_context_prototype,
    ContextPrototype,
    mean_score_over_mask,
    resize_similarity_map,
    select_feature,
    similarity_map,
)


@dataclass(frozen=True)
class PreparedReferenceGuide:
    references: tuple[ReferenceExample, ...]
    context_references: tuple[ContextReference, ...]
    embeddings: tuple[Sam3ImageEmbedding, ...]
    prototype: ContextPrototype
    concept_id: int


class ReferenceGuidedMaskGenerator:
    def __init__(
        self,
        predictor,
        *,
        base_generator=None,
        feature_layer: str | int = "image_embed",
        context_score_weight: float = 1.0,
        base_score_weight: float = 0.1,
        negative_context_mode: str = "local",
        negative_context_weight: float = 0.75,
        negative_context_scale: float = 2.0,
        min_context_score: float | None = None,
        max_masks: int | None = None,
    ) -> None:
        if context_score_weight < 0:
            raise ValueError("context_score_weight must be non-negative")
        if base_score_weight < 0:
            raise ValueError("base_score_weight must be non-negative")
        if max_masks is not None and max_masks <= 0:
            raise ValueError("max_masks must be positive")
        self.predictor = predictor
        self.base_generator = base_generator
        self.context_score_weight = float(context_score_weight)
        self.base_score_weight = float(base_score_weight)
        self.min_context_score = min_context_score
        self.max_masks = max_masks
        self.feature_layer = feature_layer
        self.negative_context_mode = negative_context_mode
        self.negative_context_weight = float(negative_context_weight)
        self.negative_context_scale = float(negative_context_scale)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        **kwargs,
    ) -> "ReferenceGuidedMaskGenerator":
        return cls(
            predictor=Sam3Predictor.from_checkpoint(path, device=device),
            **kwargs,
        )

    @torch.inference_mode()
    def prepare_references(
        self,
        references: Sequence[ReferenceExample],
    ) -> PreparedReferenceGuide:
        if not references:
            raise ValueError("references must be non-empty")
        concept_id = _shared_reference_concept_id(references)
        context_references = tuple(_context_references(references))
        embeddings = tuple(
            self.predictor.encode_image_batch(
                [reference.image for reference in context_references]
            )
        )
        prototype = build_context_prototype(
            context_references,
            embeddings,
            feature_layer=self.feature_layer,
            negative_context_mode=self.negative_context_mode,
            negative_context_scale=self.negative_context_scale,
        )
        return PreparedReferenceGuide(
            references=tuple(references),
            context_references=context_references,
            embeddings=embeddings,
            prototype=prototype,
            concept_id=concept_id,
        )

    @torch.inference_mode()
    def generate(
        self,
        target_image: Image.Image | np.ndarray,
        references: Sequence[ReferenceExample] | PreparedReferenceGuide,
        *,
        max_masks: int | None = None,
    ) -> list[MaskInstance]:
        if self.base_generator is None:
            raise ValueError("base_generator is required for generate")
        candidates = self.base_generator.generate_instances(target_image)
        return self.rerank(
            target_image,
            candidates,
            references,
            max_masks=max_masks,
        )

    @torch.inference_mode()
    def rerank(
        self,
        target_image: Image.Image | np.ndarray,
        candidates: Sequence[MaskInstance],
        references: Sequence[ReferenceExample] | PreparedReferenceGuide,
        *,
        max_masks: int | None = None,
    ) -> list[MaskInstance]:
        if isinstance(references, PreparedReferenceGuide):
            prepared = references
        elif not references:
            raise ValueError("references must be non-empty")
        if not candidates:
            return []

        if isinstance(references, PreparedReferenceGuide):
            target_embedding = self.predictor.encode_image_batch([target_image])[0]
        else:
            context_references = _context_references(references)
            image_batch = [reference.image for reference in context_references] + [
                target_image
            ]
            embeddings = self.predictor.encode_image_batch(image_batch)
            prepared = self._prepare_from_embeddings(
                references,
                context_references,
                embeddings[:-1],
            )
            target_embedding = embeddings[-1]

        target_features = select_feature(target_embedding, self.feature_layer)
        similarity = similarity_map(
            target_features,
            prepared.prototype,
            negative_context_weight=self.negative_context_weight,
        )
        similarity_full = resize_similarity_map(
            similarity,
            target_embedding.orig_hw,
        )
        target_image_size = (target_embedding.orig_hw[1], target_embedding.orig_hw[0])

        ranked: list[MaskInstance] = []
        for candidate in candidates:
            if candidate.image_size != target_image_size:
                raise ValueError("candidate image_size must match target image size")
            full_mask = candidate.to_full_mask()
            context_score = mean_score_over_mask(similarity_full, full_mask)
            if self.min_context_score is not None and context_score < float(
                self.min_context_score
            ):
                continue
            ranked.append(
                _reranked_instance(
                    candidate,
                    concept_id=prepared.concept_id,
                    context_score=context_score,
                    combined_score=(
                        context_score * self.context_score_weight
                        + candidate.score * self.base_score_weight
                    ),
                )
            )

        ranked.sort(key=lambda item: item.score, reverse=True)
        limit = self.max_masks if max_masks is None else max_masks
        if limit is not None:
            if limit <= 0:
                raise ValueError("max_masks must be positive")
            ranked = ranked[: int(limit)]
        return ranked

    def _prepare_from_embeddings(
        self,
        references: Sequence[ReferenceExample],
        context_references: Sequence[ContextReference],
        embeddings: Sequence[Sam3ImageEmbedding],
    ) -> PreparedReferenceGuide:
        concept_id = _shared_reference_concept_id(references)
        prototype = build_context_prototype(
            context_references,
            embeddings,
            feature_layer=self.feature_layer,
            negative_context_mode=self.negative_context_mode,
            negative_context_scale=self.negative_context_scale,
        )
        return PreparedReferenceGuide(
            references=tuple(references),
            context_references=tuple(context_references),
            embeddings=tuple(embeddings),
            prototype=prototype,
            concept_id=concept_id,
        )


def _context_references(
    references: Sequence[ReferenceExample],
) -> list[ContextReference]:
    context_references: list[ContextReference] = []
    for reference in references:
        if reference.image is None:
            raise ValueError("reference image is required")
        context_references.append(
            ContextReference(
                image=reference.image,
                mask=reference.mask,
                weight=reference.weight,
            )
        )
    return context_references


def _shared_reference_concept_id(references: Sequence[ReferenceExample]) -> int:
    concept_id = references[0].concept_id
    if any(reference.concept_id != concept_id for reference in references):
        raise ValueError("references must share the same concept_id")
    return int(concept_id)


def _reranked_instance(
    candidate: MaskInstance,
    *,
    concept_id: int,
    context_score: float,
    combined_score: float,
) -> MaskInstance:
    return MaskInstance(
        segmentation=candidate.segmentation,
        bbox=candidate.bbox,
        area=candidate.area,
        score=combined_score,
        source="reference_guided",
        concept_id=concept_id,
        object_id=candidate.object_id,
        context_score=context_score,
        base_score=candidate.score,
        predicted_iou=candidate.predicted_iou,
        stability_score=candidate.stability_score,
        point_coords=candidate.point_coords,
        crop_box=candidate.crop_box,
        crop_grid=candidate.crop_grid,
        crop_index=candidate.crop_index,
        image_size=candidate.image_size,
    )

from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

from ...model.grounding.prompt import Prompt
from .types import PreparedVisualConcept, PreparedVisualPrompts, VisualExemplar


class VisualPromptEncoder:
    def __init__(
        self,
        grounding_model: torch.nn.Module,
        *,
        device: torch.device | str = "cuda",
        image_size: int = 1008,
        mask_prompt_encoder: torch.nn.Module | None = None,
    ) -> None:
        self.model = grounding_model
        self.device = torch.device(device)
        self.mask_prompt_encoder = mask_prompt_encoder
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(int(image_size), int(image_size))),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    @torch.inference_mode()
    def prepare(self, exemplars: Sequence[VisualExemplar]) -> PreparedVisualPrompts:
        if not exemplars:
            raise ValueError("exemplars must be non-empty")

        grouped: dict[int, list[VisualExemplar]] = {}
        for exemplar in exemplars:
            grouped.setdefault(int(exemplar.concept_id), []).append(exemplar)

        concepts: list[PreparedVisualConcept] = []
        for concept_id, group in grouped.items():
            embeds = []
            masks = []
            for exemplar in group:
                embed, mask = self._encode_exemplar(exemplar)
                embeds.append(embed)
                masks.append(mask)
            concepts.append(
                PreparedVisualConcept(
                    concept_id=concept_id,
                    exemplars=tuple(group),
                    visual_prompt_embed=torch.cat(embeds, dim=0),
                    visual_prompt_mask=torch.cat(masks, dim=1),
                )
            )
        return PreparedVisualPrompts(concepts=tuple(concepts))

    def preprocess_image(
        self,
        image: Image.Image | np.ndarray | torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        if isinstance(image, Image.Image):
            width, height = image.size
        elif isinstance(image, np.ndarray):
            height, width = image.shape[-3:-1] if image.ndim == 3 else image.shape[-2:]
        elif isinstance(image, torch.Tensor):
            height, width = image.shape[-2:]
        else:
            raise TypeError("image must be a PIL image, NumPy array, or tensor")

        image_tensor = v2.functional.to_image(image).to(self.device)
        image_tensor = self.transform(image_tensor).unsqueeze(0)
        return image_tensor, (int(height), int(width))

    def _encode_exemplar(
        self,
        exemplar: VisualExemplar,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor, _ = self.preprocess_image(exemplar.image)
        backbone_out = self.model.backbone.forward_image(image_tensor)
        image_features, image_pos_embeds, image_sizes = _select_image_features(
            backbone_out,
            num_feature_levels=getattr(self.model, "num_feature_levels", 1),
        )
        mask = _mask_tensor(
            exemplar.mask,
            target_hw=image_tensor.shape[-2:],
            device=self.device,
        )
        prompt = Prompt(
            mask_embeddings=mask.view(1, 1, 1, *mask.shape[-2:]),
            mask_mask=torch.zeros(1, 1, dtype=torch.bool, device=self.device),
            mask_labels=torch.ones(1, 1, dtype=torch.long, device=self.device),
        )

        if getattr(self.model.geometry_encoder, "mask_encoder", None) is not None:
            return self.model.geometry_encoder(
                geo_prompt=prompt,
                img_feats=image_features,
                img_sizes=image_sizes,
                img_pos_embeds=image_pos_embeds,
            )
        if self.mask_prompt_encoder is not None:
            return _encode_with_mask_prompt_encoder(
                self.mask_prompt_encoder,
                mask,
                _feature_tensor(backbone_out["backbone_fpn"][-1]),
            )
        raise RuntimeError(
            "visual prompt mask encoding requires a geometry mask_encoder "
            "or mask_prompt_encoder"
        )


def _select_image_features(
    backbone_out: dict[str, object],
    *,
    num_feature_levels: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[tuple[int, int]]]:
    features = [_feature_tensor(item) for item in backbone_out["backbone_fpn"]]
    pos_embeds = [_feature_tensor(item) for item in backbone_out["vision_pos_enc"]]
    features = features[-num_feature_levels:]
    pos_embeds = pos_embeds[-num_feature_levels:]
    image_sizes = [tuple(pos.shape[-2:]) for pos in pos_embeds]
    image_features = [feat.flatten(2).permute(2, 0, 1) for feat in features]
    image_pos_embeds = [pos.flatten(2).permute(2, 0, 1) for pos in pos_embeds]
    return image_features, image_pos_embeds, image_sizes


def _feature_tensor(value) -> torch.Tensor:
    return getattr(value, "tensors", value)


def _mask_tensor(
    mask: np.ndarray | torch.Tensor,
    *,
    target_hw: tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    mask_array = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
    mask_array = np.asarray(mask_array).astype(bool, copy=False)
    target_h, target_w = (int(target_hw[0]), int(target_hw[1]))
    resized = Image.fromarray(mask_array.astype(np.uint8) * 255).resize(
        (target_w, target_h),
        resample=Image.Resampling.NEAREST,
    )
    mask_array = np.asarray(resized) > 0
    return torch.as_tensor(mask_array, dtype=torch.float32, device=device).view(
        1,
        1,
        target_h,
        target_w,
    )


def _encode_with_mask_prompt_encoder(
    mask_prompt_encoder: torch.nn.Module,
    mask: torch.Tensor,
    image_features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        encoded = mask_prompt_encoder(
            pix_feat=image_features,
            masks=mask,
            skip_mask_sigmoid=True,
        )
    except TypeError:
        encoded = mask_prompt_encoder(masks=mask, pix_feat=image_features)

    if isinstance(encoded, dict):
        features = encoded["vision_features"]
        pos = encoded["vision_pos_enc"][0]
    else:
        features, pos = encoded
    tokens = (features + pos).flatten(2).permute(2, 0, 1)
    token_mask = torch.zeros(
        tokens.shape[1],
        tokens.shape[0],
        dtype=torch.bool,
        device=tokens.device,
    )
    return tokens, token_mask

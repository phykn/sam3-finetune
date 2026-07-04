from contextlib import nullcontext
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..checkpoint import LoadReport
from ..transforms import Sam3Transforms
from .builder import build_model
from .types import Sam3ImageEmbedding as _Sam3ImageEmbedding
from .types import Sam3PromptBatch as _Sam3PromptBatch


class Sam3Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
        load_report: LoadReport | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.transforms = Sam3Transforms(resolution=1008, mask_threshold=0.0)
        self.load_report = load_report
        self._embedding: _Sam3ImageEmbedding | None = None

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "Sam3Predictor":
        model, report = build_model(str(checkpoint_path), device=device)
        return cls(model=model, device=device, load_report=report)

    @torch.inference_mode()
    def set_image(self, image: Image.Image | np.ndarray) -> None:
        self.set_image_embedding(self.encode_image(image))

    def set_image_embedding(self, embedding: _Sam3ImageEmbedding) -> None:
        self._embedding = embedding

    def encode_image_tensor_batch(
        self,
        input_tensor: torch.Tensor,
        orig_hws: Sequence[tuple[int, int]],
        *,
        inference: bool = True,
    ) -> list[_Sam3ImageEmbedding]:
        if input_tensor.ndim != 4 or input_tensor.shape[0] == 0:
            raise ValueError("input_tensor must be a non-empty BCHW batch")
        if input_tensor.shape[0] != len(orig_hws):
            raise ValueError("orig_hws length must match batch size")

        context = torch.inference_mode() if inference else nullcontext()
        with context:
            features = self.model.encode_image(input_tensor.to(self.device))

        image_embed = features["image_embed"]
        high_res_features = tuple(features["high_res_features"])
        embeddings: list[_Sam3ImageEmbedding] = []
        for index, orig_hw in enumerate(orig_hws):
            embeddings.append(
                _Sam3ImageEmbedding(
                    image_embed=image_embed[index : index + 1],
                    high_res_features=tuple(
                        feature[index : index + 1] for feature in high_res_features
                    ),
                    orig_hw=orig_hw,
                )
            )
        return embeddings

    def encode_image(
        self,
        image: Image.Image | np.ndarray,
        *,
        inference: bool = True,
    ) -> _Sam3ImageEmbedding:
        return self.encode_image_batch([image], inference=inference)[0]

    def encode_image_batch(
        self,
        images: Sequence[Image.Image | np.ndarray],
        *,
        inference: bool = True,
    ) -> list[_Sam3ImageEmbedding]:
        if not images:
            raise ValueError("images batch must be non-empty")
        tensors: list[torch.Tensor] = []
        orig_hws: list[tuple[int, int]] = []
        for image in images:
            tensor, orig_hw = self.transforms.preprocess_image(image, self.device)
            tensors.append(tensor)
            orig_hws.append(orig_hw)
        input_tensor = torch.cat(tensors, dim=0)
        return self.encode_image_tensor_batch(
            input_tensor,
            orig_hws,
            inference=inference,
        )

    @torch.inference_mode()
    def predict(
        self,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._embedding is None:
            raise RuntimeError("Call set_image() before predict().")
        return self.predict_from_embedding(
            self._embedding,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

    @torch.inference_mode()
    def predict_from_embedding(
        self,
        embedding: _Sam3ImageEmbedding,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        concat_points, mask_prompt = self._prepare_prompt_tensors(
            embedding,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
        )
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_prompt,
        )
        low_res_masks, iou_predictions, _tokens, _obj_scores = self.model.mask_decoder(
            image_embeddings=embedding.image_embed,
            image_pe=self.model.prompt_encoder.get_dense_pe().to(self.device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=True,
            high_res_features=list(embedding.high_res_features),
        )
        return self._format_outputs(
            low_res_masks,
            iou_predictions,
            embedding.orig_hw,
            return_logits=return_logits,
            squeeze_batch=True,
        )

    @torch.inference_mode()
    def predict_from_embedding_batches(
        self,
        prompt_batches: Sequence[_Sam3PromptBatch],
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if not prompt_batches:
            raise ValueError("prompt_batches must be non-empty")

        sparse_parts: list[torch.Tensor] = []
        dense_parts: list[torch.Tensor] = []
        image_parts: list[torch.Tensor] = []
        high_res_parts: list[list[torch.Tensor]] | None = None
        split_sizes: list[int] = []
        orig_hws: list[tuple[int, int]] = []
        embeddings: list[_Sam3ImageEmbedding] = []
        sparse_token_count: int | None = None

        for prompt_batch in prompt_batches:
            embedding = prompt_batch.embedding
            embeddings.append(embedding)
            concat_points, mask_prompt = self._prepare_prompt_tensors(
                embedding,
                point_coords=prompt_batch.point_coords,
                point_labels=prompt_batch.point_labels,
                box=prompt_batch.box,
                mask_input=prompt_batch.mask_input,
            )
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=concat_points,
                boxes=None,
                masks=mask_prompt,
            )
            if sparse_token_count is None:
                sparse_token_count = sparse_embeddings.shape[1]
            elif sparse_embeddings.shape[1] != sparse_token_count:
                raise ValueError(
                    "All prompt batches must have the same sparse token count"
                )

            batch_size = sparse_embeddings.shape[0]
            if embedding.image_embed.shape[0] != 1:
                raise ValueError("Each Sam3ImageEmbedding must contain one image")

            sparse_parts.append(sparse_embeddings)
            dense_parts.append(dense_embeddings)
            image_parts.append(embedding.image_embed.expand(batch_size, -1, -1, -1))
            if high_res_parts is None:
                high_res_parts = [[] for _ in embedding.high_res_features]
            if len(embedding.high_res_features) != len(high_res_parts):
                raise ValueError("All embeddings must have the same feature levels")
            for feature_index, feature in enumerate(embedding.high_res_features):
                high_res_parts[feature_index].append(
                    feature.expand(batch_size, -1, -1, -1)
                )
            split_sizes.append(batch_size)
            orig_hws.append(embedding.orig_hw)

        assert high_res_parts is not None
        same_embedding = all(embedding is embeddings[0] for embedding in embeddings)
        if same_embedding:
            image_embeddings = embeddings[0].image_embed
            high_res_features = list(embeddings[0].high_res_features)
            repeat_image = True
        else:
            image_embeddings = torch.cat(image_parts, dim=0)
            high_res_features = [
                torch.cat(feature_parts, dim=0) for feature_parts in high_res_parts
            ]
            repeat_image = False
        sparse_embeddings = torch.cat(sparse_parts, dim=0)
        dense_embeddings = torch.cat(dense_parts, dim=0)

        low_res_masks, iou_predictions, _tokens, _obj_scores = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe().to(self.device),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        start = 0
        for split_size, orig_hw in zip(split_sizes, orig_hws):
            end = start + split_size
            results.append(
                self._format_outputs(
                    low_res_masks[start:end],
                    iou_predictions[start:end],
                    orig_hw,
                    return_logits=return_logits,
                    squeeze_batch=False,
                )
            )
            start = end
        return results

    def _prepare_prompt_tensors(
        self,
        embedding: _Sam3ImageEmbedding,
        point_coords: np.ndarray | None = None,
        point_labels: np.ndarray | None = None,
        box: np.ndarray | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        concat_points = None
        if point_coords is not None:
            if point_labels is None:
                raise ValueError("point_labels must be supplied with point_coords")
            coords = self.transforms.transform_coords(
                point_coords, embedding.orig_hw
            ).to(self.device)
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if coords.ndim == 2:
                coords = coords[None, ...]
                labels = labels[None, ...]
            concat_points = (coords, labels)

        if box is not None:
            box_coords = self.transforms.transform_box(box, embedding.orig_hw).to(
                self.device
            )
            box_labels = torch.tensor([2, 3], dtype=torch.int, device=self.device)
            box_labels = box_labels.expand(box_coords.shape[0], 2)
            if concat_points is None:
                concat_points = (box_coords, box_labels)
            else:
                concat_points = (
                    torch.cat([box_coords, concat_points[0]], dim=1),
                    torch.cat([box_labels, concat_points[1]], dim=1),
                )

        mask_prompt = None
        if mask_input is not None:
            mask_prompt = torch.as_tensor(
                mask_input,
                dtype=torch.float32,
                device=self.device,
            )
            if mask_prompt.ndim == 2:
                mask_prompt = mask_prompt[None, None, :, :]
            elif mask_prompt.ndim == 3:
                mask_prompt = mask_prompt[:, None, :, :]
            elif mask_prompt.ndim != 4:
                raise ValueError("mask_input must have 2, 3, or 4 dimensions")
            if mask_prompt.shape[-2:] != self.model.prompt_encoder.mask_input_size:
                mask_prompt = F.interpolate(
                    mask_prompt,
                    size=self.model.prompt_encoder.mask_input_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )

        if concat_points is None and mask_prompt is None:
            raise ValueError("Provide at least one point, box, or mask prompt.")
        if concat_points is None and mask_prompt is not None:
            concat_points = (
                torch.zeros(mask_prompt.shape[0], 1, 2, device=self.device),
                -torch.ones(
                    mask_prompt.shape[0],
                    1,
                    dtype=torch.int,
                    device=self.device,
                ),
            )
        return concat_points, mask_prompt

    def _format_outputs(
        self,
        low_res_masks: torch.Tensor,
        iou_predictions: torch.Tensor,
        orig_hw: tuple[int, int],
        *,
        return_logits: bool,
        squeeze_batch: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        masks = self.transforms.postprocess_masks(
            low_res_masks,
            orig_hw,
            return_logits=return_logits,
        )
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0).float()
        if squeeze_batch:
            masks = masks.squeeze(0)
            iou_predictions = iou_predictions.squeeze(0)
            low_res_masks = low_res_masks.squeeze(0)
        return (
            masks.detach().cpu().numpy(),
            iou_predictions.float().detach().cpu().numpy(),
            low_res_masks.detach().cpu().numpy(),
        )

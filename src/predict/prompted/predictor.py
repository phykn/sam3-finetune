from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ... import types as api_types
from ...model.build import build_model
from .prompts import prepare_prompt_tensors
from .transforms import ImageTransforms


class Sam3Predictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.transforms = ImageTransforms(resolution=1008, mask_threshold=0.0)
        self._dense_pe: torch.Tensor | None = None

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
    ) -> "Sam3Predictor":
        model = build_model(str(path), device=device)
        return cls(model=model.image, device=device)

    def dense_pe(self) -> torch.Tensor:
        if self._dense_pe is None or self._dense_pe.device != self.device:
            self._dense_pe = self.model.prompt_encoder.get_dense_pe().to(self.device)
        return self._dense_pe

    def encode_image_tensor_batch(
        self,
        input_tensor: torch.Tensor,
        orig_hws: Sequence[tuple[int, int]],
        *,
        inference: bool = True,
    ) -> list[api_types.Sam3ImageEmbedding]:
        if input_tensor.ndim != 4 or input_tensor.shape[0] == 0:
            raise ValueError("input_tensor must be a non-empty BCHW batch")
        if input_tensor.shape[0] != len(orig_hws):
            raise ValueError("orig_hws length must match batch size")

        context = torch.inference_mode() if inference else nullcontext()
        with context:
            features = self.model.encode_image(input_tensor.to(self.device))

        image_embed = features["image_embed"]
        high_res_features = tuple(features["high_res_features"])
        embeddings: list[api_types.Sam3ImageEmbedding] = []
        for index, orig_hw in enumerate(orig_hws):
            embeddings.append(
                api_types.Sam3ImageEmbedding(
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
    ) -> api_types.Sam3ImageEmbedding:
        return self.encode_image_batch([image], inference=inference)[0]

    def encode_image_batch(
        self,
        images: Sequence[Image.Image | np.ndarray],
        *,
        inference: bool = True,
    ) -> list[api_types.Sam3ImageEmbedding]:
        if not images:
            raise ValueError("images batch must be non-empty")
        input_tensor, orig_hws = self.transforms.preprocess_images(
            tuple(images),
            self.device,
        )
        return self.encode_image_tensor_batch(
            input_tensor,
            orig_hws,
            inference=inference,
        )

    @torch.inference_mode()
    def predict_from_embedding(
        self,
        embedding: api_types.Sam3ImageEmbedding,
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
        box: np.ndarray | torch.Tensor | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        low_res_masks, iou_predictions = self.decode_low_res_from_embedding(
            embedding,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )
        return self._format_outputs(
            low_res_masks,
            iou_predictions,
            embedding.orig_hw,
            return_logits=return_logits,
            squeeze_batch=True,
        )

    @torch.inference_mode()
    def decode_low_res_from_embedding(
        self,
        embedding: api_types.Sam3ImageEmbedding,
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
        box: np.ndarray | torch.Tensor | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
        multimask_output: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            image_pe=self.dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=True,
            high_res_features=list(embedding.high_res_features),
        )
        return low_res_masks, iou_predictions

    @torch.inference_mode()
    def predict_from_embedding_batches(
        self,
        prompt_batches: Sequence[api_types.Sam3PromptBatch],
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        low_res_masks, iou_predictions, split_sizes, orig_hws = (
            self._decode_low_res_prompt_batches(
                prompt_batches,
                multimask_output=multimask_output,
            )
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

    @torch.inference_mode()
    def decode_low_res_from_embedding_batches(
        self,
        prompt_batches: Sequence[api_types.Sam3PromptBatch],
        multimask_output: bool = True,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        low_res_masks, iou_predictions, split_sizes, _orig_hws = (
            self._decode_low_res_prompt_batches(
                prompt_batches,
                multimask_output=multimask_output,
            )
        )

        results: list[tuple[torch.Tensor, torch.Tensor]] = []
        start = 0
        for split_size in split_sizes:
            end = start + split_size
            results.append((low_res_masks[start:end], iou_predictions[start:end]))
            start = end
        return results

    def postprocess_low_res_masks(
        self,
        low_res_masks: torch.Tensor,
        orig_hw: tuple[int, int],
        *,
        return_logits: bool = False,
    ) -> np.ndarray:
        masks = torch.as_tensor(low_res_masks, device=self.device)
        if masks.ndim == 3:
            masks = masks[:, None, :, :]
        if masks.ndim != 4:
            raise ValueError("low_res_masks must have shape BxMxHxW or BxHxW")
        masks = self.transforms.postprocess_masks(
            masks,
            orig_hw,
            return_logits=return_logits,
        )
        return masks.detach().cpu().numpy()

    def _decode_low_res_prompt_batches(
        self,
        prompt_batches: Sequence[api_types.Sam3PromptBatch],
        *,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[tuple[int, int]]]:
        if not prompt_batches:
            raise ValueError("prompt_batches must be non-empty")

        (
            sparse_embeddings,
            dense_embeddings,
            split_sizes,
            orig_hws,
            embeddings,
        ) = self._encode_prompt_batches(prompt_batches)
        image_parts: list[torch.Tensor] = []
        high_res_parts: list[list[torch.Tensor]] | None = None
        for embedding, split_size in zip(embeddings, split_sizes):
            if embedding.image_embed.shape[0] != 1:
                raise ValueError("Each Sam3ImageEmbedding must contain one image")
            image_parts.append(embedding.image_embed.expand(split_size, -1, -1, -1))
            if high_res_parts is None:
                high_res_parts = [[] for _ in embedding.high_res_features]
            if len(embedding.high_res_features) != len(high_res_parts):
                raise ValueError("All embeddings must have the same feature levels")
            for feature_index, feature in enumerate(embedding.high_res_features):
                high_res_parts[feature_index].append(
                    feature.expand(split_size, -1, -1, -1)
                )

        if high_res_parts is None:
            high_res_parts = []
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

        low_res_masks, iou_predictions, _tokens, _obj_scores = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        return low_res_masks, iou_predictions, split_sizes, orig_hws

    def _encode_prompt_batches(
        self,
        prompt_batches: Sequence[api_types.Sam3PromptBatch],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        list[tuple[int, int]],
        list[api_types.Sam3ImageEmbedding],
    ]:
        if self._can_concat_point_prompts(prompt_batches):
            return self._encode_concat_point_prompts(prompt_batches)

        sparse_parts: list[torch.Tensor] = []
        dense_parts: list[torch.Tensor] = []
        split_sizes: list[int] = []
        orig_hws: list[tuple[int, int]] = []
        embeddings: list[api_types.Sam3ImageEmbedding] = []
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

            sparse_parts.append(sparse_embeddings)
            dense_parts.append(dense_embeddings)
            split_sizes.append(sparse_embeddings.shape[0])
            orig_hws.append(embedding.orig_hw)

        return (
            torch.cat(sparse_parts, dim=0),
            torch.cat(dense_parts, dim=0),
            split_sizes,
            orig_hws,
            embeddings,
        )

    def _can_concat_point_prompts(
        self,
        prompt_batches: Sequence[api_types.Sam3PromptBatch],
    ) -> bool:
        return all(
            prompt_batch.point_coords is not None
            and prompt_batch.point_labels is not None
            and prompt_batch.box is None
            and prompt_batch.mask_input is None
            for prompt_batch in prompt_batches
        )

    def _encode_concat_point_prompts(
        self,
        prompt_batches: Sequence[api_types.Sam3PromptBatch],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int],
        list[tuple[int, int]],
        list[api_types.Sam3ImageEmbedding],
    ]:
        coord_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        split_sizes: list[int] = []
        orig_hws: list[tuple[int, int]] = []
        embeddings: list[api_types.Sam3ImageEmbedding] = []

        for prompt_batch in prompt_batches:
            embedding = prompt_batch.embedding
            coords = self.transforms.transform_coords(
                prompt_batch.point_coords,
                embedding.orig_hw,
            ).to(self.device)
            labels = torch.as_tensor(
                prompt_batch.point_labels,
                dtype=torch.int,
                device=self.device,
            )
            if coords.ndim == 2:
                coords = coords[None, ...]
                labels = labels[None, ...]
            if coords.ndim != 3 or labels.ndim != 2:
                raise ValueError("point prompts must have shape BxNx2 and BxN")
            if coords.shape[:2] != labels.shape:
                raise ValueError("point coordinates and labels must align")
            coord_parts.append(coords)
            label_parts.append(labels)
            split_sizes.append(coords.shape[0])
            orig_hws.append(embedding.orig_hw)
            embeddings.append(embedding)

        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=(torch.cat(coord_parts, dim=0), torch.cat(label_parts, dim=0)),
            boxes=None,
            masks=None,
        )
        return sparse_embeddings, dense_embeddings, split_sizes, orig_hws, embeddings

    def _prepare_prompt_tensors(
        self,
        embedding: api_types.Sam3ImageEmbedding,
        point_coords: np.ndarray | torch.Tensor | None = None,
        point_labels: np.ndarray | torch.Tensor | None = None,
        box: np.ndarray | torch.Tensor | None = None,
        mask_input: np.ndarray | torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        return prepare_prompt_tensors(
            transforms=self.transforms,
            prompt_encoder=self.model.prompt_encoder,
            device=self.device,
            embedding=embedding,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
        )

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

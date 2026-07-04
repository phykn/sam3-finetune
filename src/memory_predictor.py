from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import torch
from PIL import Image

from .checkpoint import LoadReport
from .video_builder import build_video_memory_model


@dataclass(frozen=True)
class Sam3MemoryReference:
    image: Image.Image | np.ndarray
    mask: np.ndarray | torch.Tensor
    obj_id: int


@dataclass(frozen=True)
class PreparedMemoryReference:
    reference: Sam3MemoryReference
    frame_index: int


@dataclass(frozen=True)
class Sam3MemoryPrediction:
    frame_index: int
    obj_ids: list[int]
    masks: np.ndarray
    scores: np.ndarray


class Sam3MemoryPredictor:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
        load_report: LoadReport | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.load_report = load_report
        self.image_size = int(getattr(model, "image_size", 1008))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device | str = "cuda",
        multiplex_count: int = 16,
        max_num_objects: int = 16,
    ) -> "Sam3MemoryPredictor":
        model, report = build_video_memory_model(
            checkpoint_path=checkpoint_path,
            device=device,
            multiplex_count=multiplex_count,
            max_num_objects=max_num_objects,
        )
        return cls(model=model, device=device, load_report=report)

    @staticmethod
    def prepare_references(
        references: Sequence[Sam3MemoryReference],
    ) -> list[PreparedMemoryReference]:
        if not references:
            raise ValueError("references must be non-empty")
        return [
            PreparedMemoryReference(reference=reference, frame_index=index)
            for index, reference in enumerate(references)
        ]

    @torch.inference_mode()
    def predict(
        self,
        target_image: Image.Image | np.ndarray,
        references: Sequence[Sam3MemoryReference],
    ) -> Sam3MemoryPrediction:
        prepared = self.prepare_references(references)
        images = [item.reference.image for item in prepared] + [target_image]
        frame_tensor, orig_hw = self._preprocess_image_sequence(images)
        target_frame_index = len(prepared)

        inference_state = self.model.init_state(
            video_height=orig_hw[0],
            video_width=orig_hw[1],
            num_frames=len(images),
            cached_features=None,
            offload_video_to_cpu=True,
            offload_state_to_cpu=False,
        )
        inference_state["images"] = frame_tensor
        inference_state["orig_height"] = orig_hw[0]
        inference_state["orig_width"] = orig_hw[1]

        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with autocast_context:
            for item in prepared:
                mask = self._mask_to_tensor(item.reference.mask, orig_hw)
                self.model.add_new_masks(
                    inference_state,
                    frame_idx=item.frame_index,
                    obj_ids=[int(item.reference.obj_id)],
                    masks=mask,
                )

            self.model.propagate_in_video_preflight(
                inference_state,
                run_mem_encoder=True,
            )

            final_output = None
            for output in self.model.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=target_frame_index + 1,
                reverse=False,
                tqdm_disable=True,
                run_mem_encoder=True,
            ):
                if output[0] == target_frame_index:
                    final_output = output
                    break

        if final_output is None:
            raise RuntimeError("target frame was not produced by propagation")

        frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores = final_output
        masks = (video_res_masks > 0).detach().cpu().numpy()
        scores = obj_scores.detach().float().cpu().numpy()
        return Sam3MemoryPrediction(
            frame_index=int(frame_idx),
            obj_ids=[int(obj_id) for obj_id in obj_ids],
            masks=masks,
            scores=scores,
        )

    def _preprocess_image_sequence(
        self,
        images: Sequence[Image.Image | np.ndarray],
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        pil_images = [self._to_pil(image) for image in images]
        widths = {image.width for image in pil_images}
        heights = {image.height for image in pil_images}
        if len(widths) != 1 or len(heights) != 1:
            raise ValueError("all reference and target images must share one size")
        orig_hw = (pil_images[0].height, pil_images[0].width)

        tensors = []
        mean = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float16)[:, None, None]
        std = torch.tensor((0.5, 0.5, 0.5), dtype=torch.float16)[:, None, None]
        for image in pil_images:
            resized = image.resize((self.image_size, self.image_size))
            array = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1).to(torch.float16)
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=0), orig_hw

    def _mask_to_tensor(
        self,
        mask: np.ndarray | torch.Tensor,
        orig_hw: tuple[int, int],
    ) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.detach().to(dtype=torch.float32)
        else:
            mask_tensor = torch.from_numpy(np.asarray(mask)).to(dtype=torch.float32)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError("reference mask must have shape HxW or 1xHxW")
        if tuple(mask_tensor.shape[-2:]) != orig_hw:
            raise ValueError("reference mask size must match image size")
        return mask_tensor

    @staticmethod
    def _to_pil(image: Image.Image | np.ndarray) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")

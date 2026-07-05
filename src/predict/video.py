from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ..model.build import build_model
from ..types import MemoryPrediction, MemoryReference
from .image_transform import preprocess_rgb_images, scale_coords, to_rgb_pil


@dataclass(frozen=True)
class PreparedReference:
    reference: MemoryReference
    frame_index: int


class VideoMemoryInference:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.image_size = int(getattr(model, "image_size", 1008))

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cuda",
        multiplex_count: int = 16,
        max_num_objects: int = 16,
    ) -> "VideoMemoryInference":
        model = build_model(
            path=path,
            device=device,
            multiplex_count=multiplex_count,
            max_num_objects=max_num_objects,
        )
        return cls(model=model.video, device=device)

    @staticmethod
    def prepare_references(
        references: Sequence[MemoryReference],
    ) -> list[PreparedReference]:
        if not references:
            raise ValueError("references must be non-empty")
        return [
            PreparedReference(reference=reference, frame_index=index)
            for index, reference in enumerate(references)
        ]

    @torch.inference_mode()
    def predict(
        self,
        target_image: Image.Image | np.ndarray,
        references: Sequence[MemoryReference],
        target_point_coords: np.ndarray | torch.Tensor | None = None,
        target_point_labels: np.ndarray | torch.Tensor | None = None,
        target_obj_id: int | None = None,
        target_point_mode: str = "interaction",
    ) -> MemoryPrediction:
        if target_point_mode not in {"interaction", "memory"}:
            raise ValueError("target_point_mode must be 'interaction' or 'memory'")
        if references:
            prepared = self.prepare_references(references)
        elif target_point_coords is not None:
            prepared = []
        else:
            raise ValueError("references must be non-empty")
        images = [item.reference.image for item in prepared] + [target_image]
        target_frame_index = len(prepared)
        frame_tensor, orig_hw, frame_hws = self.preprocess_image_sequence(
            images,
            output_image_index=target_frame_index,
        )

        inference_state = self.model.init_state(
            video_height=orig_hw[0],
            video_width=orig_hw[1],
            num_frames=len(images),
            cached_features=None,
            device=self.device,
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
                mask = self.mask_to_tensor(
                    item.reference.mask,
                    source_hw=frame_hws[item.frame_index],
                    target_hw=orig_hw,
                )
                self.model.add_new_masks(
                    inference_state,
                    frame_idx=item.frame_index,
                    obj_ids=[int(item.reference.obj_id)],
                    masks=mask,
                )

            use_memory_target_point = (
                target_point_coords is not None
                and target_point_mode == "memory"
                and bool(prepared)
            )
            if target_point_mode == "memory" and target_point_coords is not None:
                if not prepared:
                    raise ValueError(
                        "target_point_mode='memory' requires at least one reference"
                    )
                if target_point_labels is None:
                    raise ValueError(
                        "target_point_labels must be supplied with target_point_coords"
                    )

            if target_point_coords is not None and not use_memory_target_point:
                if target_point_labels is None:
                    raise ValueError(
                        "target_point_labels must be supplied with target_point_coords"
                    )
                if target_obj_id is not None:
                    point_obj_id = int(target_obj_id)
                elif prepared:
                    point_obj_id = int(prepared[0].reference.obj_id)
                else:
                    point_obj_id = 1
                point_coords, point_labels = self.target_points_to_tensors(
                    target_point_coords,
                    target_point_labels,
                    target_hw=orig_hw,
                )
                self.model.add_new_points(
                    inference_state=inference_state,
                    frame_idx=target_frame_index,
                    obj_id=point_obj_id,
                    points=point_coords,
                    labels=point_labels,
                    clear_old_points=True,
                    rel_coordinates=False,
                )

            self.model.propagate_in_video_preflight(
                inference_state,
                run_mem_encoder=True,
            )

            if use_memory_target_point:
                assert target_point_labels is not None
                point_obj_id = (
                    int(target_obj_id)
                    if target_obj_id is not None
                    else int(prepared[0].reference.obj_id)
                )
                target_result = self.predict_target_points_with_memory(
                    inference_state=inference_state,
                    frame_idx=target_frame_index,
                    obj_id=point_obj_id,
                    target_hw=orig_hw,
                    point_coords=target_point_coords,
                    point_labels=target_point_labels,
                )
                return target_result

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
        return MemoryPrediction(
            frame_index=int(frame_idx),
            obj_ids=[int(obj_id) for obj_id in obj_ids],
            masks=masks,
            scores=scores,
        )

    def predict_target_points_with_memory(
        self,
        inference_state: dict,
        frame_idx: int,
        obj_id: int,
        target_hw: tuple[int, int],
        point_coords: np.ndarray | torch.Tensor,
        point_labels: np.ndarray | torch.Tensor,
    ) -> MemoryPrediction:
        if obj_id not in inference_state["obj_id_to_idx"]:
            raise ValueError(f"target obj_id {obj_id} is not present in references")
        obj_idx = int(inference_state["obj_id_to_idx"][obj_id])
        points, labels = self.target_points_to_tensors(
            point_coords,
            point_labels,
            target_hw=target_hw,
        )
        point_inputs = {
            "point_coords": points.to(self.device),
            "point_labels": labels.to(self.device),
        }
        current_out, pred_masks = self.model._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=inference_state["output_dict"],
            frame_idx=frame_idx,
            batch_size=self.model._get_obj_num(inference_state),
            is_init_cond_frame=False,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=False,
            run_mem_encoder=False,
            objects_to_interact=[obj_idx],
        )
        _low_res_masks, video_res_masks = self.model._get_orig_video_res_output(
            inference_state,
            pred_masks,
        )
        masks = (video_res_masks > 0).detach().cpu().numpy()
        scores = current_out["object_score_logits"].detach().float().cpu().numpy()
        return MemoryPrediction(
            frame_index=int(frame_idx),
            obj_ids=[int(obj_id) for obj_id in inference_state["obj_ids"]],
            masks=masks,
            scores=scores,
        )

    def preprocess_image_sequence(
        self,
        images: Sequence[Image.Image | np.ndarray],
        output_image_index: int = -1,
    ) -> tuple[torch.Tensor, tuple[int, int], list[tuple[int, int]]]:
        return preprocess_rgb_images(
            list(images),
            resolution=self.image_size,
            output_image_index=output_image_index,
            dtype=torch.float16,
        )

    def mask_to_tensor(
        self,
        mask: np.ndarray | torch.Tensor,
        source_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if isinstance(mask, torch.Tensor):
            mask_tensor = mask.detach().to(dtype=torch.float32)
        else:
            mask_tensor = torch.from_numpy(np.asarray(mask)).to(dtype=torch.float32)
        if mask_tensor.ndim == 2:
            mask_tensor = mask_tensor.unsqueeze(0)
        if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
            raise ValueError("reference mask must have shape HxW or 1xHxW")
        if tuple(mask_tensor.shape[-2:]) != source_hw:
            raise ValueError("reference mask size must match image size")
        if source_hw != target_hw:
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=target_hw,
                mode="nearest",
            )[0]
        return mask_tensor

    def target_points_to_tensors(
        self,
        point_coords: np.ndarray | torch.Tensor,
        point_labels: np.ndarray | torch.Tensor,
        target_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coords = torch.as_tensor(point_coords, dtype=torch.float32)
        labels = torch.as_tensor(point_labels, dtype=torch.int64)
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        if labels.ndim == 1:
            labels = labels.unsqueeze(0)
        if coords.ndim != 3 or coords.shape[-1] != 2:
            raise ValueError("target_point_coords must have shape Nx2 or BxNx2")
        if labels.ndim != 2:
            raise ValueError("target_point_labels must have shape N or BxN")
        if coords.shape[:2] != labels.shape:
            raise ValueError("target point coordinates and labels must align")

        return scale_coords(coords, target_hw, self.image_size), labels

    @staticmethod
    def to_pil(image: Image.Image | np.ndarray) -> Image.Image:
        return to_rgb_pil(image)

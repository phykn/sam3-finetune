from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch import nn

from ...data import image as image_data


def start(
    model: nn.Module,
    image: Image.Image | np.ndarray,
    mask: np.ndarray | torch.Tensor,
    obj_id: int,
    device: str | torch.device,
    offload_video_to_cpu: bool = False,
    offload_state_to_cpu: bool = False,
) -> dict[str, object]:
    tensor, features, orig_hw = cache_frame(model, image, device)
    state = model.init_state(
        video_height=orig_hw[0],
        video_width=orig_hw[1],
        num_frames=1,
        cached_features={0: (tensor, features)},
        device=device,
        offload_video_to_cpu=offload_video_to_cpu,
        offload_state_to_cpu=offload_state_to_cpu,
    )
    state["offload_video_to_cpu"] = bool(offload_video_to_cpu)
    state["offload_state_to_cpu"] = bool(offload_state_to_cpu)
    state["images"] = [tensor[0].detach().cpu()] if offload_video_to_cpu else []
    model.add_masks(
        state,
        frame_idx=0,
        obj_ids=[obj_id],
        masks=mask_tensor(mask, device),
    )
    model.propagate_in_video_preflight(state, run_mem_encoder=True)
    return {"state": state, "obj_id": obj_id, "next_frame": 1}


def add_masks(
    model: nn.Module,
    session_state: dict[str, object],
    masks: np.ndarray | torch.Tensor,
    obj_ids: list[int],
    device: str | torch.device,
    frame_idx: int | None = None,
) -> list[int]:
    tracker_state = session_state["state"]
    pending_state = _copy_state(tracker_state)
    if frame_idx is None:
        frame_idx = session_state["next_frame"] - 1
    restore_frame(model, pending_state, frame_idx, device)

    _, ids, _, _ = model.add_masks(
        pending_state,
        frame_idx,
        obj_ids,
        mask_tensor(masks, device),
    )
    model.propagate_in_video_preflight(pending_state, run_mem_encoder=True)
    trim_frame_cache(pending_state, session_state["next_frame"] - 1)
    ids = list(ids)
    _commit_state(tracker_state, pending_state)
    return ids


def remove_objects(
    model: nn.Module,
    session_state: dict[str, object],
    obj_ids: list[int],
    strict: bool = True,
) -> list[int]:
    tracker_state = session_state["state"]
    pending_state = _copy_state(tracker_state)
    ids, _ = model.remove_objects(
        pending_state,
        obj_ids,
        strict=strict,
    )
    ids = list(ids)
    _commit_state(tracker_state, pending_state)
    return ids


def predict(
    model: nn.Module,
    state: dict[str, object],
    image: Image.Image | np.ndarray,
    device: str | torch.device,
    threshold: float,
) -> dict[str, object]:
    frame_idx = state["next_frame"]
    tracker_state = state["state"]
    image = image_data.convert_rgb(image)
    orig_hw = (image.height, image.width)
    expected_hw = (tracker_state["video_height"], tracker_state["video_width"])
    if orig_hw != expected_hw:
        raise ValueError(
            f"video frame size must stay {expected_hw[1]}x{expected_hw[0]}"
        )
    pending_state = _copy_state(tracker_state)
    tensor, features, _ = cache_frame(model, image, device)
    pending_state["cached_features"][frame_idx] = (tensor, features)
    if pending_state["offload_video_to_cpu"]:
        pending_state["images"].append(tensor[0].detach().cpu())
    pending_state["num_frames"] = max(pending_state["num_frames"], frame_idx + 1)

    result = None
    for result in model.propagate_in_video(
        pending_state,
        start_frame_idx=frame_idx,
        max_frame_num_to_track=1,
        tqdm_disable=True,
        run_mem_encoder=True,
    ):
        pass

    if result is None:
        raise RuntimeError("video tracker returned no frame output")
    output = format_output(result, threshold)
    trim_frame_cache(pending_state, frame_idx)
    _commit_state(tracker_state, pending_state)
    state["next_frame"] = frame_idx + 1
    return output


def _copy_state(value, memo=None):
    if memo is None:
        memo = {}
    value_id = id(value)
    if value_id in memo:
        return memo[value_id]
    if isinstance(value, torch.Tensor):
        memo[value_id] = value
        return value
    if isinstance(value, dict):
        copied = value.copy()
        memo[value_id] = copied
        for key, item in value.items():
            copied[key] = _copy_state(item, memo)
        return copied
    if isinstance(value, list):
        copied = []
        memo[value_id] = copied
        copied.extend(_copy_state(item, memo) for item in value)
        return copied
    if isinstance(value, tuple):
        copied = tuple(_copy_state(item, memo) for item in value)
        memo[value_id] = copied
        return copied
    if isinstance(value, set):
        copied = set()
        memo[value_id] = copied
        copied.update(_copy_state(item, memo) for item in value)
        return copied
    return deepcopy(value, memo)


def _commit_state(state: dict[str, object], pending: dict[str, object]) -> None:
    state.clear()
    state.update(pending)


def restore_frame(
    model: nn.Module,
    state: dict[str, object],
    frame_idx: int,
    device: str | torch.device,
) -> None:
    if frame_idx in state["cached_features"]:
        return
    images = state.get("images", [])
    if not state.get("offload_video_to_cpu") or not 0 <= frame_idx < len(images):
        raise KeyError(f"frame {frame_idx} is not cached")

    tensor = images[frame_idx].to(device).float().unsqueeze(0)
    state["cached_features"][frame_idx] = (
        tensor,
        encode_frame(model, tensor),
    )


def trim_frame_cache(state: dict[str, object], keep: int) -> None:
    if not state.get("offload_video_to_cpu"):
        return
    cache = state["cached_features"]
    for frame_idx in list(cache):
        if frame_idx != keep:
            del cache[frame_idx]


def cache_frame(
    model: nn.Module,
    image: Image.Image | np.ndarray,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, object], tuple[int, int]]:
    tensor, orig_hw = image_data.make_tensor(image, model.image_size, device)
    return tensor, encode_frame(model, tensor), orig_hw


def encode_frame(model: nn.Module, tensor: torch.Tensor) -> dict[str, object]:
    return model.forward_image(
        tensor,
        need_sam3_out=False,
        need_interactive_out=True,
        need_propagation_out=True,
    )


def mask_tensor(
    mask: np.ndarray | torch.Tensor, device: str | torch.device
) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        mask = mask.to(device=device, dtype=torch.float32)
    else:
        mask = torch.as_tensor(np.asarray(mask), dtype=torch.float32, device=device)
    if mask.ndim == 2:
        mask = mask[None]
    if mask.ndim != 3:
        raise ValueError("video reference mask must have shape HxW or NxHxW")
    return mask


def format_output(
    result: tuple[int, list[int], torch.Tensor, torch.Tensor, torch.Tensor],
    threshold: float,
) -> dict[str, object]:
    frame_idx, obj_ids, low_res, video_res, scores = result
    masks = (video_res[:, 0] > threshold).detach().cpu().numpy()
    return {
        "frame_idx": frame_idx,
        "obj_ids": list(obj_ids),
        "masks": masks,
        "scores": scores.reshape(-1).float().detach().cpu().numpy(),
        "logits": low_res[:, 0].float().detach().cpu().numpy(),
        "raw": {
            "low_res_masks": low_res,
            "video_res_masks": video_res,
            "obj_scores": scores,
        },
    }

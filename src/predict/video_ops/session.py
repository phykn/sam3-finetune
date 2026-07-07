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
) -> dict[str, object]:
    tensor, features, orig_hw = cache_frame(model, image, device)
    state = model.init_state(
        video_height=orig_hw[0],
        video_width=orig_hw[1],
        num_frames=1,
        cached_features={0: (tensor, features)},
        device=device,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    )
    model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[obj_id],
        masks=mask_tensor(mask, device),
    )
    model.propagate_in_video_preflight(state, run_mem_encoder=True)
    return {"state": state, "obj_id": obj_id, "next_frame": 1}


def predict(
    model: nn.Module,
    state: dict[str, object],
    image: Image.Image | np.ndarray,
    device: str | torch.device,
    threshold: float,
) -> dict[str, object]:
    frame_idx = state["next_frame"]
    tensor, features, _ = cache_frame(model, image, device)
    tracker_state = state["state"]
    tracker_state["cached_features"][frame_idx] = (tensor, features)
    tracker_state["num_frames"] = max(tracker_state["num_frames"], frame_idx + 1)

    result = None
    for result in model.propagate_in_video(
        tracker_state,
        start_frame_idx=frame_idx,
        max_frame_num_to_track=1,
        reverse=False,
        tqdm_disable=True,
        run_mem_encoder=True,
    ):
        pass

    state["next_frame"] = frame_idx + 1
    if result is None:
        raise RuntimeError("video tracker returned no frame output")
    return format_output(result, threshold)


def cache_frame(
    model: nn.Module,
    image: Image.Image | np.ndarray,
    device: str | torch.device,
) -> tuple[torch.Tensor, dict[str, object], tuple[int, int]]:
    tensor, orig_hw = image_data.make_tensor(image, model.image_size, device)
    features = model.forward_image(
        tensor,
        need_sam3_out=False,
        need_interactive_out=True,
        need_propagation_out=True,
    )
    return tensor, features, orig_hw


def mask_tensor(
    mask: np.ndarray | torch.Tensor, device: str | torch.device
) -> torch.Tensor:
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

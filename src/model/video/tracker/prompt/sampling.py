import numpy as np
import torch
from numpy.typing import NDArray


def edt_triton(data: torch.Tensor) -> torch.Tensor:
    """CPU/OpenCV fallback for upstream's Triton EDT helper.

    The tracker imports this function eagerly upstream. Keeping a local fallback
    lets Windows import the memory stack without requiring Triton.
    """
    assert data.dim() == 3
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "opencv-python is required for EDT fallback when triton is unavailable"
        ) from exc

    outputs = []
    for item in data.detach().cpu().numpy().astype("uint8"):
        outputs.append(cv2.distanceTransform(item, cv2.DIST_L2, 0))
    output = torch.from_numpy(np.stack(outputs, axis=0))
    return output.to(device=data.device, dtype=torch.float32)


def sample_box_points(
    masks: torch.Tensor,
    noise: float = 0.1,
    noise_bound: int = 20,
    top_left_label: int = 2,
    bottom_right_label: int = 3,
) -> tuple[NDArray, NDArray]:
    device = masks.device
    box_coords = mask_to_box(masks)
    batch_size, _, height, width = masks.shape
    box_labels = torch.tensor(
        [top_left_label, bottom_right_label],
        dtype=torch.int,
        device=device,
    ).repeat(batch_size)

    if noise > 0.0:
        if not isinstance(noise_bound, torch.Tensor):
            noise_bound = torch.tensor(noise_bound, device=device)

        box_width = box_coords[..., 2] - box_coords[..., 0]
        box_height = box_coords[..., 3] - box_coords[..., 1]
        max_dx = torch.min(box_width * noise, noise_bound)
        max_dy = torch.min(box_height * noise, noise_bound)
        box_noise = 2 * torch.rand(batch_size, 1, 4, device=device) - 1
        box_noise = box_noise * torch.stack((max_dx, max_dy, max_dx, max_dy), dim=-1)

        box_coords = box_coords + box_noise
        image_bounds = torch.tensor([width, height, width, height], device=device) - 1
        box_coords.clamp_(torch.zeros_like(image_bounds), image_bounds)

    # Shape: B x 2 x 2, with top-left and bottom-right points.
    box_coords = box_coords.reshape(-1, 2, 2)
    box_labels = box_labels.reshape(-1, 2)
    return box_coords, box_labels


def mask_to_box(masks: torch.Tensor):
    batch_size, _, height, width = masks.shape
    device = masks.device
    mask_area = masks.sum(dim=(-1, -2))
    xs = torch.arange(width, device=device, dtype=torch.int32)
    ys = torch.arange(height, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(batch_size, 1, height, width)
    grid_ys = grid_ys[None, None, ...].expand(batch_size, 1, height, width)

    min_xs, _ = torch.min(torch.where(masks, grid_xs, width).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, height).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    box_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    return torch.where(
        mask_area[..., None] > 0, box_coords, torch.zeros_like(box_coords)
    )


def sample_random_points_from_errors(gt_masks, pred_masks, num_pt=1):
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape
    assert num_pt >= 0

    batch_size, _, height, width = gt_masks.shape
    device = gt_masks.device

    fp_masks = ~gt_masks & pred_masks
    fn_masks = gt_masks & ~pred_masks
    all_correct = torch.all((gt_masks == pred_masks).flatten(2), dim=2)
    all_correct = all_correct[..., None, None]

    # Shape: last dim stores FP and FN sampling maps.
    pts_noise = torch.rand(batch_size, num_pt, height, width, 2, device=device)
    pts_noise[..., 0] *= fp_masks | (all_correct & ~gt_masks)
    pts_noise[..., 1] *= fn_masks
    pts_idx = pts_noise.flatten(2).argmax(dim=2)
    labels = (pts_idx % 2).to(torch.int32)
    pts_idx = pts_idx // 2
    pts_x = pts_idx % width
    pts_y = pts_idx // width
    points = torch.stack([pts_x, pts_y], dim=2).to(torch.float)
    return points, labels


def sample_one_point_from_error_center(gt_masks, pred_masks, padding=True):
    if pred_masks is None:
        pred_masks = torch.zeros_like(gt_masks)
    assert gt_masks.dtype == torch.bool and gt_masks.size(1) == 1
    assert pred_masks.dtype == torch.bool and pred_masks.shape == gt_masks.shape

    batch_size, _, height, width = gt_masks.shape
    fp_masks = (~gt_masks & pred_masks).squeeze(1)
    fn_masks = (gt_masks & ~pred_masks).squeeze(1)

    if padding:
        padded_fp_masks = torch.zeros(
            batch_size,
            height + 2,
            width + 2,
            dtype=fp_masks.dtype,
            device=fp_masks.device,
        )
        padded_fp_masks[:, 1 : height + 1, 1 : width + 1] = fp_masks
        padded_fn_masks = torch.zeros(
            batch_size,
            height + 2,
            width + 2,
            dtype=fp_masks.dtype,
            device=fp_masks.device,
        )
        padded_fn_masks[:, 1 : height + 1, 1 : width + 1] = fn_masks
    else:
        padded_fp_masks = fp_masks
        padded_fn_masks = fn_masks

    fn_mask_dt = edt_triton(padded_fn_masks)
    fp_mask_dt = edt_triton(padded_fp_masks)
    if padding:
        fn_mask_dt = fn_mask_dt[:, 1:-1, 1:-1]
        fp_mask_dt = fp_mask_dt[:, 1:-1, 1:-1]

    fn_max, fn_argmax = fn_mask_dt.reshape(batch_size, -1).max(dim=-1)
    fp_max, fp_argmax = fp_mask_dt.reshape(batch_size, -1).max(dim=-1)
    is_positive = fn_max > fp_max
    chosen = torch.where(is_positive, fn_argmax, fp_argmax)
    points_x = chosen % width
    points_y = chosen // width

    labels = is_positive.long()
    points = torch.stack([points_x, points_y], -1)
    return points.unsqueeze(1), labels.unsqueeze(1)


def get_next_point(gt_masks, pred_masks, method):
    if method == "uniform":
        return sample_random_points_from_errors(gt_masks, pred_masks)
    if method == "center":
        return sample_one_point_from_error_center(gt_masks, pred_masks)
    raise ValueError(f"unknown sampling method {method}")

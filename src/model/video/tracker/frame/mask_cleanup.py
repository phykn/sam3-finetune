import numpy as np
import torch
import torch.nn.functional as F


def fill_holes_in_mask_scores(
    mask,
    max_area=None,
    fill_holes=True,
    remove_sprinkles=True,
    fill_hole_area=None,
):
    if fill_hole_area is not None and max_area is None:
        max_area = fill_hole_area

    if max_area <= 0:
        return mask

    if fill_holes:
        mask_bg = mask <= 0
        bg_area_thresh = max_area
        _, areas_bg = get_connected_components_with_padding(mask_bg)
        small_components_bg = mask_bg & (areas_bg <= bg_area_thresh)
        mask = torch.where(small_components_bg, 0.1, mask)

    if remove_sprinkles:
        mask_fg = mask > 0
        fg_area_thresh = torch.sum(mask_fg, dim=(2, 3), keepdim=True, dtype=torch.int32)
        fg_area_thresh.floor_divide_(2).clamp_(max=max_area)
        _, areas_fg = get_connected_components_with_padding(mask_fg)
        small_components_fg = mask_fg & (areas_fg <= fg_area_thresh)
        mask = torch.where(small_components_fg, -0.1, mask)

    return mask


def get_connected_components_with_padding(mask):
    mask = mask.to(torch.uint8)
    _, _, height, width = mask.shape

    pad_h = height % 2
    pad_w = width % 2
    if pad_h == 0 and pad_w == 0:
        labels, counts = connected_components(mask)
    else:
        mask_pad = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0)
        labels, counts = connected_components(mask_pad)
        labels = labels[:, :, :height, :width]
        counts = counts[:, :, :height, :width]

    return labels, counts


def connected_components(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """OpenCV connected-components fallback matching upstream return shapes."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "opencv-python is required for connected-components fallback"
        ) from exc

    if mask.dim() == 3:
        mask_3d = mask
        restore_channel = False
    else:
        assert mask.dim() == 4 and mask.shape[1] == 1
        mask_3d = mask[:, 0]
        restore_channel = True

    labels_list = []
    counts_list = []
    for item in mask_3d.detach().cpu().numpy().astype("uint8"):
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            item,
            connectivity=8,
        )
        counts = np.zeros_like(labels, dtype=np.int32)
        for label_index in range(1, num_labels):
            counts[labels == label_index] = stats[label_index, cv2.CC_STAT_AREA]
        labels_list.append(torch.from_numpy(labels.astype(np.int32)))
        counts_list.append(torch.from_numpy(counts))

    labels_t = torch.stack(labels_list, dim=0).to(mask.device)
    counts_t = torch.stack(counts_list, dim=0).to(mask.device)
    if restore_channel:
        labels_t = labels_t[:, None]
        counts_t = counts_t[:, None]
    return labels_t, counts_t

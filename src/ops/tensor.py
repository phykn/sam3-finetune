import torch
import torch.nn.functional as F


def inverse_sigmoid(tensor: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    tensor = tensor.clamp(min=0, max=1)
    numerator = tensor.clamp(min=eps)
    denominator = (1 - tensor).clamp(min=eps)
    return torch.log(numerator / denominator)


def interpolate(
    tensor: torch.Tensor,
    size: tuple[int, int] | None = None,
    scale_factor: float | tuple[float, ...] | None = None,
    mode: str = "nearest",
    align_corners: bool | None = None,
) -> torch.Tensor:
    if tensor.numel() > 0:
        return F.interpolate(tensor, size, scale_factor, mode, align_corners)

    assert (
        tensor.shape[0] != 0 or tensor.shape[1] != 0
    ), "At least one of the two first dimensions must be non zero"

    if tensor.shape[1] == 0:
        return F.interpolate(
            tensor.transpose(0, 1), size, scale_factor, mode, align_corners
        ).transpose(0, 1)

    return F.interpolate(tensor, size, scale_factor, mode, align_corners)

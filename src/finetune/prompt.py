import torch

from ..data import prompt as prompt_data


def build_prompt(
    item: dict,
    image_size: int,
    mask_size: tuple[int, int],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
    points = prompt_data.build_points(
        item["points"],
        item["point_labels"],
        (image_size, image_size),
        image_size,
        device,
    )
    box = prompt_data.build_box(
        item["box"],
        (image_size, image_size),
        image_size,
        device,
    )
    if box is None:
        point_prompt = points
    elif points is None:
        point_prompt = box
    else:
        point_prompt = (
            torch.cat([box[0], points[0]], dim=1),
            torch.cat([box[1], points[1]], dim=1),
        )

    mask_prompt = prompt_data.build_mask(item["mask"], mask_size, device)
    if point_prompt is None:
        batch = 1 if mask_prompt is None else mask_prompt.shape[0]
        point_prompt = (
            torch.zeros(batch, 1, 2, device=device),
            -torch.ones(batch, 1, dtype=torch.int, device=device),
        )
    return point_prompt, mask_prompt


def build_prompts(
    items: list[dict],
    image_size: int,
    mask_size: tuple[int, int],
    device: torch.device,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor | None]:
    prompts = [build_prompt(item, image_size, mask_size, device) for item in items]
    points = (
        torch.cat([prompt[0][0] for prompt in prompts]),
        torch.cat([prompt[0][1] for prompt in prompts]),
    )
    if prompts[0][1] is None:
        masks = None
    else:
        masks = torch.cat([prompt[1] for prompt in prompts])
    return points, masks

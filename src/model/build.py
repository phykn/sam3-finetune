import torch

from ..io.checkpoint import load_pth, remap_model
from .model import Sam3Model


def build_model(
    path: str | None = None,
    device: torch.device | str = "cuda",
    multiplex_count: int = 16,
    max_num_objects: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
) -> Sam3Model:
    model = Sam3Model(
        multiplex_count=multiplex_count,
        max_num_objects=max_num_objects,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    ).to(device=device)
    model.share()
    model.eval()

    if path is None:
        return model

    checkpoint = load_pth(path)
    state, _ignored = remap_model(checkpoint)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.share()
    model.eval()
    return model

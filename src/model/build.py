from dataclasses import dataclass

import torch

from ..io.checkpoint import load_pth, remap_model
from .sam3 import Sam3Model


@dataclass(frozen=True)
class LoadReport:
    ignored: list[str]
    missing: list[str]
    unexpected: list[str]


def build_model(
    path: str | None = None,
    device: torch.device | str = "cuda",
    multiplex_count: int = 16,
    max_num_objects: int = 16,
    use_fa3: bool = False,
    use_rope_real: bool = False,
    return_report: bool = False,
) -> Sam3Model | tuple[Sam3Model, LoadReport]:
    model = Sam3Model(
        multiplex_count=multiplex_count,
        max_num_objects=max_num_objects,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    ).to(device=device)
    model.eval()

    if path is None:
        if return_report:
            return model, LoadReport(ignored=[], missing=[], unexpected=[])
        return model

    checkpoint = load_pth(path)
    state, ignored = remap_model(checkpoint)
    load_result = model.load_state_dict(state, strict=False)
    model.to(device)
    model.share()
    model.eval()
    if return_report:
        return model, LoadReport(
            ignored=list(ignored),
            missing=list(load_result.missing_keys),
            unexpected=list(load_result.unexpected_keys),
        )
    return model

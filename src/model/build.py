from dataclasses import dataclass
from pathlib import Path

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
    include_language: bool = False,
    bpe_path: str | Path | None = None,
    return_report: bool = False,
) -> Sam3Model | tuple[Sam3Model, LoadReport]:
    if include_language:
        bpe_path = _resolve_bpe_path(bpe_path)
    model = Sam3Model(
        multiplex_count=multiplex_count,
        max_num_objects=max_num_objects,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
        include_language=include_language,
        bpe_path=None if bpe_path is None else str(bpe_path),
    ).to(device=device)
    model.eval()

    if path is None:
        if return_report:
            return model, LoadReport(ignored=[], missing=[], unexpected=[])
        return model

    checkpoint = load_pth(path)
    state, ignored = remap_model(checkpoint, include_language=include_language)
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


def _resolve_bpe_path(path: str | Path | None) -> Path:
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"BPE tokenizer file does not exist: {resolved}")
        return resolved

    root = Path(__file__).resolve().parents[2]
    candidates = [
        root / "weight" / "bpe_simple_vocab_16e6.txt.gz",
        root / "sam3-main" / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("BPE tokenizer file is required when include_language=True")

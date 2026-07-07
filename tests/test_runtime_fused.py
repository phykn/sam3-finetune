import torch
from src.ml.runtime.fused import apply_addmm_activation


def test_apply_addmm_activation_returns_bfloat16():
    linear = torch.nn.Linear(4, 8)
    tensor = torch.randn(2, 3, 4)

    with torch.inference_mode():
        out = apply_addmm_activation(torch.nn.GELU, linear, tensor)

    assert out.shape == (2, 3, 8)
    assert out.dtype == torch.bfloat16

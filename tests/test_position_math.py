import torch

from src.ml.components.nn.position import PositionEmbeddingSine
from src.ml.components.sam.rope import (
    apply_rotary_enc,
    apply_rotary_enc_real,
    compute_axial_cis,
)


def test_position_encoding_cache_uses_input_device():
    encoder = PositionEmbeddingSine(num_pos_feats=8)
    image = torch.zeros(2, 1, 8, 8)

    output = encoder(image)

    assert output.device == image.device
    assert output.shape == (2, 8, 8, 8)


def test_real_and_complex_rope_match_and_preserve_norm():
    torch.manual_seed(0)
    query = torch.randn(1, 2, 4, 8)
    key = torch.randn(1, 2, 4, 8)
    freqs = compute_axial_cis(dim=8, end_x=2, end_y=2)

    complex_query, complex_key = apply_rotary_enc(query, key, freqs)
    real_query, real_key = apply_rotary_enc_real(
        query,
        key,
        freqs.real,
        freqs.imag,
    )

    torch.testing.assert_close(real_query, complex_query)
    torch.testing.assert_close(real_key, complex_key)
    torch.testing.assert_close(complex_query.norm(dim=-1), query.norm(dim=-1))

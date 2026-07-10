import torch

from src.ml.components.nn.attention import MultiheadAttention


def test_custom_attention_matches_torch_attention():
    torch.manual_seed(0)
    custom = MultiheadAttention(8, 2, dropout=0.0).eval()
    reference = torch.nn.MultiheadAttention(8, 2, dropout=0.0).eval()
    reference.load_state_dict(custom.state_dict())
    query = torch.randn(5, 2, 8)
    padding = torch.tensor([[False, False, False, True, True]]).expand(2, -1)

    actual, _ = custom(query, query, query, key_padding_mask=padding)
    expected, _ = reference(
        query,
        query,
        query,
        key_padding_mask=padding,
        need_weights=False,
    )

    torch.testing.assert_close(actual, expected)


def test_attention_backward_is_finite():
    torch.manual_seed(0)
    attention = MultiheadAttention(8, 2, dropout=0.0)
    query = torch.randn(5, 2, 8, requires_grad=True)

    output, _ = attention(query, query, query)
    output.square().mean().backward()

    assert torch.isfinite(query.grad).all()

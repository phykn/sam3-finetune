import src.ml.components.transformer.decoder as decoder
from src.ml.components.transformer.model import Transformer
from src.ml.components.transformer.video import (
    RotaryAttention,
    VideoDecoderLayer,
    VideoTransformerEncoder,
)


def test_transformer_implementations_are_split_by_use():
    assert not hasattr(decoder, "TransformerDecoderLayerv1")
    assert not hasattr(decoder, "TransformerDecoderLayerv2")
    assert not hasattr(decoder, "TransformerEncoderCrossAttention")
    assert RotaryAttention.__module__.endswith("transformer.video")
    assert VideoDecoderLayer.__module__.endswith("transformer.video")
    assert VideoTransformerEncoder.__module__.endswith("transformer.video")
    assert Transformer.__module__.endswith("transformer.model")

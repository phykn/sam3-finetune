import numpy as np

from src.data.augment.image import pixel


def test_pixel_augment_keeps_shape_and_uint8_for_each_op(monkeypatch):
    image = np.full((8, 10, 3), 120, dtype=np.uint8)

    for op in pixel.OPS:
        monkeypatch.setattr(pixel.np.random, "choice", lambda ops, value=op: value)

        out = pixel.augment_pixel(image)

        assert out.shape == image.shape
        assert out.dtype == np.uint8


def test_pixel_dropout_adds_black_hole_when_selected(monkeypatch):
    image = np.full((40, 40, 3), 120, dtype=np.uint8)
    monkeypatch.setattr(pixel.np.random, "choice", lambda ops: "dropout")

    out = pixel.augment_pixel(image)

    assert (out == 0).any()

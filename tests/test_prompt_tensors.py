import numpy as np
import pytest
import torch
from src.predict.prompted.prompts import prepare_prompt_tensors
from src.predict.prompted.transforms import ImageTransforms
from src.types import Sam3ImageEmbedding


class FakePromptEncoder:
    mask_input_size = (4, 4)


def _embedding() -> Sam3ImageEmbedding:
    return Sam3ImageEmbedding(
        image_embed=torch.zeros(1, 1, 1, 1),
        high_res_features=(),
        orig_hw=(10, 20),
    )


def _prepare(**kwargs):
    return prepare_prompt_tensors(
        transforms=ImageTransforms(resolution=100),
        prompt_encoder=FakePromptEncoder(),
        device=torch.device("cpu"),
        embedding=_embedding(),
        **kwargs,
    )


def test_prepare_prompt_tensors_accepts_point_only_prompt():
    concat_points, mask_prompt = _prepare(
        point_coords=np.array([[2.0, 3.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
    )

    coords, labels = concat_points
    assert mask_prompt is None
    assert coords.shape == (1, 1, 2)
    assert labels.tolist() == [[1]]
    np.testing.assert_allclose(coords.numpy(), np.array([[[10.0, 30.0]]]))


def test_prepare_prompt_tensors_accepts_box_only_prompt():
    concat_points, mask_prompt = _prepare(
        box=np.array([2.0, 1.0, 6.0, 4.0], dtype=np.float32)
    )

    coords, labels = concat_points
    assert mask_prompt is None
    assert coords.shape == (1, 2, 2)
    assert labels.tolist() == [[2, 3]]
    np.testing.assert_allclose(
        coords.numpy(),
        np.array([[[10.0, 10.0], [30.0, 40.0]]]),
    )


def test_prepare_prompt_tensors_merges_box_before_point_prompt():
    concat_points, _mask_prompt = _prepare(
        point_coords=np.array([[2.0, 3.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int64),
        box=np.array([2.0, 1.0, 6.0, 4.0], dtype=np.float32),
    )

    coords, labels = concat_points
    assert coords.shape == (1, 3, 2)
    assert labels.tolist() == [[2, 3, 1]]
    np.testing.assert_allclose(
        coords.numpy(),
        np.array([[[10.0, 10.0], [30.0, 40.0], [10.0, 30.0]]]),
    )


def test_prepare_prompt_tensors_adds_dummy_negative_point_for_mask_only_prompt():
    concat_points, mask_prompt = _prepare(mask_input=np.ones((2, 2), dtype=np.float32))

    coords, labels = concat_points
    assert coords.shape == (1, 1, 2)
    assert coords.tolist() == [[[0.0, 0.0]]]
    assert labels.tolist() == [[-1]]
    assert mask_prompt.shape == (1, 1, 4, 4)


def test_prepare_prompt_tensors_requires_labels_with_points():
    with pytest.raises(ValueError, match="point_labels"):
        _prepare(point_coords=np.array([[2.0, 3.0]], dtype=np.float32))


def test_prepare_prompt_tensors_rejects_invalid_mask_input_shape():
    with pytest.raises(ValueError, match="mask_input"):
        _prepare(mask_input=np.ones((1, 1, 1, 1, 1), dtype=np.float32))

import numpy as np
import torch

from src.data import pack
from src.predict.mask import format as mask_format


def test_resize_masks_returns_bool_mask_at_original_size():
    masks = mask_format.resize_masks(torch.ones(1, 1, 2, 2), (4, 6), 0.0)

    assert masks.shape == (1, 1, 4, 6)
    assert masks.dtype == torch.bool


def test_low_result_keeps_decoder_mask_size():
    result = mask_format.make_low(
        torch.ones(1, 1, 2, 2),
        torch.tensor([[0.75]]),
        0.0,
    )

    assert result["masks"].shape == (1, 1, 2, 2)
    assert result["masks"].dtype == bool
    assert result["scores"].tolist() == [[0.75]]


def test_objects_preserve_prompt_and_candidate_axes():
    masks = torch.full((2, 2, 2, 3), -2.0)
    masks[0, 0, :, :2] = 2.0
    masks[0, 1, :, 1:] = 2.0
    masks[1, 0, 0, :] = 2.0
    scores = torch.tensor([[0.9, 0.8], [0.7, 0.6]])
    classes = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)

    objects = mask_format.make_objects(masks, scores, (4, 6), classes)

    assert len(objects) == 3
    assert [item["object_id"] for item in objects] == [1, 2, 3]
    assert [(item["prompt_index"], item["candidate_index"]) for item in objects] == [
        (0, 0),
        (0, 1),
        (1, 0),
    ]
    assert all(item["class_id"] is None for item in objects)
    assert objects[0]["roi"].dtype == np.bool_
    assert objects[0]["metrics"]["class_logits"] == [0.0, 1.0, 2.0, 3.0]
    expected = mask_format.resize_masks(masks, (4, 6), 0.0)[0, 0].numpy()
    actual = pack.full((4, 6), objects[0]["box"], objects[0]["roi"])
    np.testing.assert_array_equal(actual, expected)


def test_objects_drop_empty_masks():
    objects = mask_format.make_objects(
        torch.full((1, 1, 2, 2), -2.0),
        torch.tensor([[0.25]]),
        (4, 4),
    )

    assert objects == []

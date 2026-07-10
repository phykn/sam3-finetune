import torch
from src.predict.mask import format as mask_format


def test_resize_masks_returns_bool_mask_at_original_size():
    masks = mask_format.resize_masks(torch.ones(1, 1, 2, 2), (4, 6), 0.0)

    assert masks.shape == (1, 1, 4, 6)
    assert masks.dtype == torch.bool


def test_full_result_converts_tensors_to_numpy():
    result = mask_format.make_full(
        torch.full((1, 1, 2, 2), 40.0),
        torch.tensor([[0.75]]),
        (4, 6),
        0.0,
    )

    assert result["masks"].shape == (1, 4, 6)
    assert result["scores"].tolist() == [0.75]
    assert result["logits"].max() == 32.0


def test_low_result_keeps_decoder_mask_size():
    result = mask_format.make_low(
        torch.ones(1, 1, 2, 2),
        torch.tensor([[0.75]]),
        0.0,
    )

    assert result["masks"].shape == (1, 2, 2)
    assert result["masks"].dtype == bool
    assert result["scores"].tolist() == [0.75]


def test_class_result_converts_logits_to_probabilities():
    result = mask_format.make_classes(torch.tensor([[[2.0, -2.0]]]))

    assert result["class_logits"].shape == (1, 2)
    assert result["class_scores"].shape == (1, 2)
    torch.testing.assert_close(
        torch.from_numpy(result["class_scores"]),
        torch.from_numpy(result["class_logits"]).sigmoid(),
    )

import numpy as np
import pytest
import torch

from src.data import pack
from src.predict.ground_ops import output


def make_image():
    features = torch.tensor(
        [
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 1.0]],
            ]
        ]
    )
    return {"backbone_fpn": (features,)}


def make_decoder_out():
    return {
        "pred_logits": torch.tensor([[[4.0], [-4.0]], [[4.0], [-4.0]]]),
        "pred_boxes": torch.tensor(
            [
                [[0.25, 0.25, 0.5, 0.5], [0.75, 0.75, 0.5, 0.5]],
                [[0.75, 0.75, 0.5, 0.5], [0.25, 0.25, 0.5, 0.5]],
            ]
        ),
        "pred_masks": torch.tensor(
            [
                [
                    [[2.0, -2.0], [-2.0, -2.0]],
                    [[-2.0, -2.0], [-2.0, 2.0]],
                ],
                [
                    [[-2.0, -2.0], [-2.0, 2.0]],
                    [[2.0, -2.0], [-2.0, -2.0]],
                ],
            ]
        ),
        "raw": {"large": torch.ones(100)},
    }


def test_candidates_filter_with_low_res_tensors_before_output_resize():
    bank = {
        1: torch.tensor([[1.0, 0.0]]),
        2: torch.tensor([[0.0, 1.0]]),
    }

    items = output.candidates(
        make_decoder_out(),
        make_image(),
        np.array([1, 2]),
        bank,
        (1008, 1008),
        score_thr=0.5,
        sim_thr=0.5,
    )

    assert len(items) == 2
    assert [item["class_id"] for item in items] == [1, 2]
    for item in items:
        assert set(item) == {"class_id", "nms_box", "logit", "metrics"}
        assert isinstance(item["logit"], torch.Tensor)
        assert item["logit"].shape == (2, 2)
        assert isinstance(item["metrics"]["score"], torch.Tensor)
        assert isinstance(item["metrics"]["similarity"], torch.Tensor)
        assert item["metrics"]["similarity"].item() > 0.99
        assert item["nms_box"].device.type == "cpu"
        assert item["logit"].device.type == "cpu"
        assert item["metrics"]["score"].device.type == "cpu"
        assert item["metrics"]["similarity"].device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_candidates_offload_cuda_tensors_to_cpu():
    decoder = {
        key: value.cuda() if isinstance(value, torch.Tensor) else value
        for key, value in make_decoder_out().items()
    }
    image = {
        "backbone_fpn": tuple(value.cuda() for value in make_image()["backbone_fpn"])
    }
    bank = {
        1: torch.tensor([[1.0, 0.0]], device="cuda"),
        2: torch.tensor([[0.0, 1.0]], device="cuda"),
    }

    items = output.candidates(
        decoder,
        image,
        np.array([1, 2]),
        bank,
        (1008, 1008),
        score_thr=0.5,
        sim_thr=0.5,
    )

    assert len(items) == 2
    for item in items:
        assert item["nms_box"].device.type == "cpu"
        assert item["logit"].device.type == "cpu"
        assert item["metrics"]["score"].device.type == "cpu"
        assert item["metrics"]["similarity"].device.type == "cpu"


def make_item(class_id, score, box, mask):
    mask = torch.as_tensor(mask)
    return {
        "class_id": class_id,
        "nms_box": torch.tensor(box, dtype=torch.float32),
        "logit": torch.where(mask, 2.0, -2.0),
        "metrics": {
            "score": torch.tensor(score),
            "similarity": torch.tensor(0.9),
        },
    }


def test_finish_resizes_only_candidates_selected_by_nms(monkeypatch):
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    items = [
        make_item(1, 0.9, (0, 0, 5, 4), mask),
        make_item(1, 0.8, (0, 0, 5, 4), mask),
        make_item(2, 0.7, (0, 0, 5, 4), mask),
    ]

    calls = []
    interpolate = output.F.interpolate

    def record(value, size, **kwargs):
        calls.append((tuple(value.shape), tuple(size)))
        return interpolate(value, size, **kwargs)

    monkeypatch.setattr(output.F, "interpolate", record)
    out = output.finish(
        items,
        nms_thr=0.5,
        top_k=None,
        orig_hw=(4, 5),
        mask_batch_size=2,
        device="cpu",
    )

    assert calls == [((2, 1, 4, 5), (4, 5))]
    assert [(item["object_id"], item["class_id"]) for item in out] == [
        (1, 1),
        (2, 2),
    ]
    assert all(item["box"] == (2, 1, 4, 3) for item in out)
    assert all("nms_box" not in item for item in out)
    assert all("mask" not in item for item in out)
    assert all(item["roi"].dtype == np.bool_ for item in out)
    for item in out:
        np.testing.assert_array_equal(pack.full((4, 5), item["box"], item["roi"]), mask)


def test_finish_applies_top_k_per_class():
    masks = []
    for index in range(3):
        mask = np.zeros((4, 6), dtype=bool)
        mask[1:3, index * 2 : index * 2 + 2] = True
        masks.append(mask)
    items = [
        make_item(1, 0.7 + index * 0.1, (index * 2, 1, index * 2 + 2, 3), mask)
        for index, mask in enumerate(masks)
    ]

    out = output.finish(
        items,
        nms_thr=0.5,
        top_k=2,
        orig_hw=(4, 6),
        mask_batch_size=2,
        device="cpu",
    )

    assert [item["metrics"]["score"] for item in out] == pytest.approx([0.9, 0.8])


def test_finish_drops_mask_that_is_empty_after_final_resize():
    mask = np.array([[True, False], [False, False]])
    item = make_item(1, 0.9, (0, 0, 1, 1), mask)

    out = output.finish(
        [item],
        nms_thr=0.5,
        top_k=None,
        orig_hw=(1, 1),
        mask_batch_size=2,
        device="cpu",
    )

    assert out == []


def test_finish_resizes_selected_masks_in_bounded_batches(monkeypatch):
    items = [
        make_item(
            class_id,
            0.9,
            (0, 0, 4, 4),
            np.ones((2, 2), dtype=bool),
        )
        for class_id in range(5)
    ]
    calls = []
    interpolate = output.F.interpolate

    def record(value, size, **kwargs):
        calls.append(value.shape[0])
        return interpolate(value, size, **kwargs)

    monkeypatch.setattr(output.F, "interpolate", record)
    out = output.finish(
        items,
        nms_thr=0.5,
        top_k=None,
        orig_hw=(4, 4),
        mask_batch_size=2,
        device="cpu",
    )

    assert calls == [2, 2, 1]
    assert [item["object_id"] for item in out] == [1, 2, 3, 4, 5]
    assert [item["class_id"] for item in out] == [0, 1, 2, 3, 4]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_finish_runs_nms_on_cpu_and_interpolation_on_cuda(monkeypatch):
    mask = np.ones((2, 2), dtype=bool)
    items = [
        make_item(1, 0.9, (0, 0, 2, 2), mask),
        make_item(1, 0.8, (0, 0, 2, 2), mask),
        make_item(2, 0.7, (0, 0, 2, 2), mask),
    ]
    nms_devices = []
    interpolate_calls = []
    nms = output.nms_indices
    interpolate = output.F.interpolate

    def record_nms(boxes, scores, threshold):
        nms_devices.append((boxes.device.type, scores.device.type))
        return nms(boxes, scores, threshold)

    def record_interpolate(value, size, **kwargs):
        interpolate_calls.append((value.device.type, value.shape[0]))
        return interpolate(value, size, **kwargs)

    monkeypatch.setattr(output, "nms_indices", record_nms)
    monkeypatch.setattr(output.F, "interpolate", record_interpolate)

    out = output.finish(
        items,
        nms_thr=0.5,
        top_k=None,
        orig_hw=(4, 4),
        mask_batch_size=2,
        device="cuda",
    )

    assert nms_devices == [("cpu", "cpu"), ("cpu", "cpu")]
    assert interpolate_calls == [("cuda", 2)]
    assert [item["class_id"] for item in out] == [1, 2]
    assert all(item["roi"].dtype == np.bool_ for item in out)

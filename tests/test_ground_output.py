import numpy as np
import pytest
import torch

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


def test_candidates_filter_on_gpu_and_return_only_cpu_values():
    bank = {
        1: torch.tensor([[1.0, 0.0]]),
        2: torch.tensor([[0.0, 1.0]]),
    }

    items = output.candidates(
        make_decoder_out(),
        make_image(),
        np.array([1, 2]),
        bank,
        (2, 2),
        score_thr=0.5,
        sim_thr=0.5,
    )

    assert len(items) == 2
    assert [item["class_id"] for item in items] == [1, 2]
    for item in items:
        assert set(item) == {"class_id", "nms_box", "mask", "logit", "metrics"}
        assert isinstance(item["mask"], np.ndarray)
        assert item["mask"].dtype == np.bool_
        assert isinstance(item["logit"], np.ndarray)
        assert isinstance(item["metrics"]["score"], float)
        assert isinstance(item["metrics"]["similarity"], float)
        assert item["metrics"]["similarity"] > 0.99


def make_item(class_id, score, box, mask):
    return {
        "class_id": class_id,
        "nms_box": box,
        "mask": mask,
        "logit": np.ones((2, 2), dtype=np.float32),
        "metrics": {"score": score, "similarity": 0.9},
    }


def test_finish_removes_only_same_class_overlap():
    mask = np.zeros((4, 5), dtype=bool)
    mask[1:3, 2:4] = True
    items = [
        make_item(1, 0.9, (0, 0, 5, 4), mask),
        make_item(1, 0.8, (0, 0, 5, 4), mask),
        make_item(2, 0.7, (0, 0, 5, 4), mask),
    ]

    out = output.finish(items, nms_thr=0.5, top_k=None)

    assert [(item["object_id"], item["class_id"]) for item in out] == [
        (1, 1),
        (2, 2),
    ]
    assert all(item["box"] == (2, 1, 4, 3) for item in out)
    assert all("nms_box" not in item for item in out)


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

    out = output.finish(items, nms_thr=0.5, top_k=2)

    assert [item["metrics"]["score"] for item in out] == pytest.approx([0.9, 0.8])

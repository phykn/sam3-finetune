import importlib

import numpy as np
from src.types import MemoryPrediction


def test_next_script_defaults_to_left_to_right_helicopter_order() -> None:
    script = importlib.import_module("scripts.next")

    args = script.parse_args([])
    paths = script.resolve_paths(args)

    assert [path.name for path in paths.frames] == [
        "heli_3.jpg",
        "heli_2.jpg",
        "heli_1.jpg",
    ]
    assert paths.output.name == "next.png"
    prompt = script.describe_reference_prompt()
    assert prompt["type"] == "point_box"
    assert prompt["points"][0]["label"] == 1


def test_next_script_selects_prediction_mask_by_object_id() -> None:
    script = importlib.import_module("scripts.next")
    masks = np.zeros((2, 1, 3, 4), dtype=bool)
    masks[1, 0, 1:, 2:] = True
    prediction = MemoryPrediction(
        frame_index=2,
        obj_ids=[3, 7],
        masks=masks,
        scores=np.array([[0.25], [0.75]], dtype=np.float32),
    )

    selected = script.prediction_mask(prediction, obj_id=7)
    summary = script.summarize_prediction(prediction, obj_id=7)

    np.testing.assert_array_equal(selected, masks[1, 0])
    assert summary["frame_index"] == 2
    assert summary["obj_id"] == 7
    assert summary["area"] == 4
    assert summary["score"] == 0.75

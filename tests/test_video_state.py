import pytest
from inspect import signature
import torch
from torch import nn

from src.ml.model.video.state import (
    add_object,
    cached_frame,
    create_state,
    forward_frames,
)
from src.ml.model.video.objects import remove_objects
from src.ml.model.video.runtime import VideoRuntime


def make_state(**kwargs):
    return create_state(
        num_frames=kwargs.get("num_frames", 3),
        video_height=5,
        video_width=7,
        cached_features={0: ("image", "features")},
        device="cpu",
    )


def test_state_registers_unique_ordered_ids():
    state = make_state()

    assert add_object(state, 8) == 0
    assert add_object(state, 3) == 1
    assert add_object(state, 8) == 0
    assert state["obj_ids"] == [8, 3]
    assert "point_inputs_per_obj" not in state


def test_cached_frame_rejects_missing_index():
    with pytest.raises(KeyError, match="frame 1 is not cached"):
        cached_frame(make_state(), 1)


def test_forward_frames_never_returns_future_or_reverse_indices():
    assert list(forward_frames(start=2, count=3, num_frames=7)) == [2, 3, 4]
    assert list(forward_frames(start=5, count=4, num_frames=7)) == [5, 6]


def test_forward_frames_rejects_empty_count():
    with pytest.raises(ValueError, match="count must be positive"):
        forward_frames(0, 0, 3)


def test_video_propagation_has_no_reverse_parameter():
    from src.ml.model.video.propagate import propagate

    assert "reverse" not in signature(propagate).parameters


def test_removing_all_objects_resets_object_state():
    state = make_state()
    add_object(state, 4)
    add_object(state, 7)
    state["mask_inputs_per_obj"][0][0] = "mask-4"
    state["mask_inputs_per_obj"][1][0] = "mask-7"
    state["multiplex_state"] = object()

    ids, updated = remove_objects(None, state, [4, 7])

    assert ids == []
    assert updated == []
    assert state["multiplex_state"] is None
    assert state["mask_inputs_per_obj"] == {}
    assert state["output_dict"]["cond_frame_outputs"] == {}


def test_strict_removal_rejects_unknown_id():
    state = make_state()
    add_object(state, 4)

    with pytest.raises(ValueError, match="Object id 9 does not exist"):
        remove_objects(None, state, [9], strict=True)


def test_per_object_output_skips_ids_missing_from_historical_frame():
    runtime = VideoRuntime.__new__(VideoRuntime)
    nn.Module.__init__(runtime)
    runtime.use_memory_selection = False
    state = {
        "obj_id_to_idx": {1: 0, 3: 1},
        "output_dict_per_obj": {
            0: {"non_cond_frame_outputs": {}},
            1: {"non_cond_frame_outputs": {}},
        },
    }
    current = {
        "pred_masks": torch.ones(1, 1, 2, 2),
        "object_score_logits": torch.ones(1, 1),
        "local_obj_id_to_idx": {1: 0},
    }

    runtime._add_output_per_object(
        state,
        frame_idx=0,
        current_out=current,
        storage_key="non_cond_frame_outputs",
    )

    assert 0 in state["output_dict_per_obj"][0]["non_cond_frame_outputs"]
    assert state["output_dict_per_obj"][1]["non_cond_frame_outputs"] == {}

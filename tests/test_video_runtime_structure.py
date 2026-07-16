from inspect import signature
from pathlib import Path

from src.ml.model.video.runtime import VideoRuntime
from src.ml.components.video.tracker.frame.inference import run_single_frame_inference
from src.ml.components.video.tracker.memory.conditioning import (
    prepare_memory_conditioned_features,
)
from src.ml.components.video.tracker.memory.context import collect_memory_context
from src.ml import structures
from src.predict.video import VideoPredictor

ROOT = Path(__file__).resolve().parents[1] / "src" / "ml"


def test_video_runtime_has_only_inference_dependencies():
    params = list(signature(VideoRuntime).parameters)
    assert params == [
        "features",
        "memory",
        "tracking",
        "multiplex_controller",
    ]
    assert not hasattr(VideoRuntime, "forward_tracking")
    assert not hasattr(VideoRuntime, "prepare_prompt_inputs")


def test_video_runtime_call_stack_has_no_point_or_reverse_parameters():
    removed = {
        "point_inputs",
        "gt_masks",
        "frames_to_add_correction_pt",
        "track_in_reverse",
        "reverse",
        "prev_sam_mask_logits",
        "objects_to_interact",
    }
    assert removed.isdisjoint(signature(VideoRuntime.track_step).parameters)
    assert removed.isdisjoint(signature(run_single_frame_inference).parameters)
    assert removed.isdisjoint(signature(prepare_memory_conditioned_features).parameters)
    assert removed.isdisjoint(signature(collect_memory_context).parameters)


def test_stateful_video_flow_lives_in_model_package():
    assert (ROOT / "model" / "video" / "runtime.py").exists()
    assert (ROOT / "model" / "video" / "state.py").exists()
    assert not (ROOT / "components" / "video" / "tracker" / "tracking.py").exists()


def test_video_predictor_exposes_dynamic_mask_api():
    assert list(signature(VideoPredictor.add_masks).parameters) == [
        "self",
        "state",
        "masks",
        "obj_ids",
        "frame_idx",
    ]
    assert list(signature(VideoPredictor.remove_objects).parameters) == [
        "self",
        "state",
        "obj_ids",
        "strict",
    ]


def test_video_training_and_point_modules_are_removed():
    tracker = ROOT / "components" / "video" / "tracker"
    removed = (
        tracker / "runtime" / "compile.py",
        tracker / "runtime" / "correction.py",
        tracker / "runtime" / "loop.py",
        tracker / "prompt" / "inputs.py",
        tracker / "prompt" / "order.py",
        tracker / "prompt" / "sampling.py",
        tracker / "prompt" / "utils.py",
        tracker / "interaction" / "points.py",
        tracker / "interaction" / "point_output.py",
        tracker / "interaction" / "point_refinement.py",
        tracker / "interaction" / "point_setup.py",
        tracker / "interaction" / "extract.py",
        tracker / "interaction" / "frame_merge.py",
        tracker / "interaction" / "merge.py",
        tracker / "interaction" / "merge_state.py",
        tracker / "interaction" / "tensor.py",
        tracker / "frame" / "mask_cleanup.py",
        tracker / "state.py",
    )
    assert not [path for path in removed if path.exists()]


def test_video_training_datapoints_are_removed():
    assert not hasattr(structures, "FindStage")
    assert not hasattr(structures, "BatchedDatapoint")

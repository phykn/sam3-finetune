import numpy as np
import torch
from PIL import Image


def _tiny_image() -> Image.Image:
    return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))


def _tiny_mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return mask


class _TinyMuxTransformer(torch.nn.Module):
    def forward(self, src, pos_src, tokens):
        return tokens, src.flatten(2).transpose(1, 2)


def test_video_memory_public_api_imports() -> None:
    import src.predict as video
    from src.model.build import build_model
    from src.predict import MemoryReference, VideoMemoryInference

    assert VideoMemoryInference.__name__ == "VideoMemoryInference"
    assert MemoryReference.__name__ == "MemoryReference"
    assert MemoryReference.__module__ == "src.predict.video_types"
    assert callable(build_model)
    assert video.VideoMemoryInference is VideoMemoryInference
    assert video.MemoryReference is MemoryReference
    assert not hasattr(video, "__all__")


def test_multiplex_mask_decoder_dynamic_single_mask_output() -> None:
    from src.model.video.tracker.decoder.multiplex import MultiplexMaskDecoder

    decoder = MultiplexMaskDecoder(
        transformer_dim=8,
        transformer=_TinyMuxTransformer(),
        multiplex_count=2,
        dynamic_multimask_via_stability=True,
    )
    decoder.eval()

    out = decoder(
        image_embeddings=torch.zeros(1, 8, 2, 2),
        image_pe=torch.zeros(1, 8, 2, 2),
        multimask_output=False,
    )

    assert out["masks"].shape == (1, 2, 1, 8, 8)
    assert out["iou_pred"].shape == (1, 2, 1)
    assert out["sam_tokens_out"].shape == (1, 2, 1, 8)


def test_multiplex_mask_decoder_shared_attribute_tokens() -> None:
    from src.model.video.tracker.decoder.multiplex import MultiplexMaskDecoder

    decoder = MultiplexMaskDecoder(
        transformer_dim=8,
        transformer=_TinyMuxTransformer(),
        multiplex_count=2,
        pred_obj_scores=True,
        decode_mask_attribute_with_shared_tokens=True,
        use_multimask_token_for_obj_ptr=True,
    )

    out = decoder(
        image_embeddings=torch.zeros(1, 8, 2, 2),
        image_pe=torch.zeros(1, 8, 2, 2),
        multimask_output=True,
    )

    assert out["masks"].shape == (1, 2, 3, 8, 8)
    assert out["iou_pred"].shape == (1, 2, 3)
    assert out["sam_tokens_out"].shape == (1, 2, 3, 8)
    assert out["object_score_logits"].shape == (1, 2, 1)


def test_consolidation_preserves_late_object_after_frame_copy() -> None:
    from src.model.video.tracker.consolidation.merge import (
        consolidate_temp_output_across_obj,
    )

    class Model:
        low_res_mask_size = 4
        use_memory_selection = False

        def _get_obj_num(self, inference_state) -> int:
            return 1

    frame_out = {
        "pred_masks": torch.ones(1, 1, 4, 4),
        "obj_ptr": torch.zeros(1, 8),
        "object_score_logits": torch.zeros(1, 1),
    }
    late_obj_out = {
        "pred_masks": torch.full((1, 1, 4, 4), 7.0),
        "object_score_logits": torch.ones(1, 1),
    }
    state = {
        "device": torch.device("cpu"),
        "storage_device": torch.device("cpu"),
        "point_inputs_per_obj": {},
        "mask_inputs_per_obj": {},
        "output_dict": {
            "cond_frame_outputs": {0: frame_out},
            "non_cond_frame_outputs": {},
        },
        "temp_output_dict_per_obj": {
            2: {
                "cond_frame_outputs": {0: late_obj_out},
                "non_cond_frame_outputs": {},
            },
        },
        "output_dict_per_obj": {
            2: {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        },
    }

    out = consolidate_temp_output_across_obj(
        Model(),
        state,
        frame_idx=0,
        is_cond=True,
        run_mem_encoder=False,
    )

    assert out["pred_masks"].shape == (3, 1, 4, 4)
    assert torch.equal(out["pred_masks"][2], late_obj_out["pred_masks"][0])


def test_video_memory_modules_import_without_triton() -> None:
    import src.model.video.tracker.frame.mask_cleanup
    import src.model.video.tracker.memory.context
    import src.model.video.tracker.model
    import src.model.video.tracker.prompt.sampling
    import src.model.video.tracker.tracking

    assert hasattr(src.model.video.tracker.memory.context, "select_closest_cond_frames")
    assert hasattr(src.model.video.tracker.prompt.sampling, "get_next_point")
    assert hasattr(
        src.model.video.tracker.frame.mask_cleanup, "fill_holes_in_mask_scores"
    )
    assert hasattr(src.model.video.tracker.tracking, "VideoTrackingDynamicMultiplex")
    assert hasattr(src.model.video.tracker.model, "Sam3VideoTrackingMultiplexDemo")


def test_video_modules_live_under_role_packages() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    tracker_root = root / "src" / "model" / "video" / "tracker"

    assert not (root / "src" / "model" / "video" / "_tracker").exists()
    assert not (root / "src" / "model" / "video" / "builder.py").exists()
    assert not (root / "src" / "model" / "video" / "checkpoint.py").exists()
    assert (root / "src" / "model" / "video" / "model.py").is_file()
    assert (root / "src" / "predict" / "video.py").is_file()
    assert (root / "src" / "predict" / "video_types.py").is_file()

    for path in (
        "consolidation/merge.py",
        "decoder/heads.py",
        "decoder/multiplex.py",
        "frame/features.py",
        "frame/mask_cleanup.py",
        "interaction/masks.py",
        "memory/conditioning.py",
        "memory/context.py",
        "memory/encoder.py",
        "memory/encoding.py",
        "multiplex/assignments.py",
        "multiplex/state.py",
        "prompt/inputs.py",
        "prompt/sampling.py",
        "runtime/init.py",
        "runtime/params.py",
        "runtime/step.py",
        "tracking.py",
        "model.py",
    ):
        assert (tracker_root / path).is_file()

    for filename in (
        "decoder.py",
        "memory.py",
        "multiplex.py",
        "multiplex_mask_decoder.py",
        "prompting.py",
        "tracker_utils.py",
    ):
        assert not (tracker_root / filename).exists()

    for filename in (
        "video_builder.py",
        "video_checkpoint.py",
        "memory_predictor.py",
        "memory.py",
        "decoder_memory.py",
        "multiplex_utils.py",
        "multiplex_mask_decoder.py",
        "tracker_utils.py",
        "video_tracking_multiplex.py",
        "video_tracking_multiplex_demo.py",
    ):
        assert not (root / "src" / filename).exists()
    assert not (root / "src" / "video" / "memory_inference.py").exists()
    assert not (root / "src" / "video").exists()


def test_memory_references_preserve_order_for_same_object_id() -> None:
    from src.predict import MemoryReference, VideoMemoryInference

    references = [
        MemoryReference(image=_tiny_image(), mask=_tiny_mask(), obj_id=3),
        MemoryReference(image=_tiny_image(), mask=_tiny_mask(), obj_id=3),
        MemoryReference(image=_tiny_image(), mask=_tiny_mask(), obj_id=9),
    ]

    prepared = VideoMemoryInference.prepare_references(references)

    assert [item.frame_index for item in prepared] == [0, 1, 2]
    assert [item.reference.obj_id for item in prepared] == [3, 3, 9]


def test_preprocess_sequence_uses_target_size_for_mixed_image_sizes() -> None:
    from src.predict import VideoMemoryInference

    predictor = object.__new__(VideoMemoryInference)
    predictor.image_size = 16
    reference = Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8))
    target = Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8))

    batch, orig_hw, frame_hws = predictor.preprocess_image_sequence(
        [reference, target],
        output_image_index=1,
    )

    assert tuple(batch.shape) == (2, 3, 16, 16)
    assert orig_hw == (12, 20)
    assert frame_hws == [(8, 10), (12, 20)]


def test_mask_to_tensor_resizes_reference_mask_to_target_size() -> None:
    from src.predict import VideoMemoryInference

    predictor = object.__new__(VideoMemoryInference)
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:7] = True

    resized = predictor.mask_to_tensor(
        mask,
        source_hw=(8, 10),
        target_hw=(12, 20),
    )

    assert tuple(resized.shape) == (1, 12, 20)
    assert resized.dtype == torch.float32
    assert resized.sum() > mask.sum()


def test_memory_predictor_adds_target_points_after_reference_masks() -> None:
    from src.predict import MemoryReference, VideoMemoryInference

    class FakeVideoModel:
        image_size = 16

        def __init__(self) -> None:
            self.added_points = None

        def init_state(self, **kwargs):
            return {"device": torch.device("cpu"), **kwargs}

        def add_new_masks(self, *_args, **kwargs) -> None:
            pass

        def add_new_points(self, *_args, **kwargs) -> None:
            self.added_points = kwargs

        def propagate_in_video_preflight(self, *_args, **kwargs) -> None:
            pass

        def propagate_in_video(self, *_args, **kwargs):
            yield (
                1,
                [7],
                None,
                torch.ones(1, 1, 12, 20),
                torch.tensor([[0.5]]),
            )

    model = FakeVideoModel()
    predictor = object.__new__(VideoMemoryInference)
    predictor.model = model
    predictor.device = torch.device("cpu")
    predictor.image_size = 16
    reference = MemoryReference(
        image=Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)),
        mask=np.ones((8, 10), dtype=bool),
        obj_id=7,
    )
    target = Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8))

    predictor.predict(
        target_image=target,
        references=[reference],
        target_point_coords=np.array([[10, 6]], dtype=np.float32),
        target_point_labels=np.array([1], dtype=np.int64),
    )

    assert model.added_points["frame_idx"] == 1
    assert model.added_points["obj_id"] == 7
    assert model.added_points["clear_old_points"] is True
    assert model.added_points["rel_coordinates"] is False
    torch.testing.assert_close(
        model.added_points["points"],
        torch.tensor([[[8.0, 8.0]]], dtype=torch.float32),
    )
    torch.testing.assert_close(
        model.added_points["labels"],
        torch.tensor([[1]], dtype=torch.int64),
    )


def test_memory_predictor_allows_target_points_without_references() -> None:
    from src.predict import VideoMemoryInference

    class FakeVideoModel:
        image_size = 16

        def __init__(self) -> None:
            self.added_masks = 0
            self.added_points = None

        def init_state(self, **kwargs):
            return {"device": torch.device("cpu"), **kwargs}

        def add_new_masks(self, *_args, **kwargs) -> None:
            self.added_masks += 1

        def add_new_points(self, *_args, **kwargs) -> None:
            self.added_points = kwargs

        def propagate_in_video_preflight(self, *_args, **kwargs) -> None:
            pass

        def propagate_in_video(self, *_args, **kwargs):
            yield (
                0,
                [5],
                None,
                torch.ones(1, 1, 12, 20),
                torch.tensor([[0.75]]),
            )

    model = FakeVideoModel()
    predictor = object.__new__(VideoMemoryInference)
    predictor.model = model
    predictor.device = torch.device("cpu")
    predictor.image_size = 16
    target = Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8))

    prediction = predictor.predict(
        target_image=target,
        references=[],
        target_point_coords=np.array([[10, 6]], dtype=np.float32),
        target_point_labels=np.array([1], dtype=np.int64),
        target_obj_id=5,
    )

    assert model.added_masks == 0
    assert model.added_points["frame_idx"] == 0
    assert model.added_points["obj_id"] == 5
    assert prediction.frame_index == 0
    assert prediction.obj_ids == [5]


def test_memory_predictor_can_combine_reference_memory_with_target_points() -> None:
    from src.predict import MemoryReference, VideoMemoryInference

    class FakeVideoModel:
        image_size = 16

        def __init__(self) -> None:
            self.added_points = None
            self.run_single_frame_kwargs = None

        def init_state(self, **kwargs):
            return {
                "device": torch.device("cpu"),
                "obj_id_to_idx": {7: 0},
                "obj_ids": [7],
                "output_dict": {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
                **kwargs,
            }

        def add_new_masks(self, inference_state, frame_idx, obj_ids, masks) -> None:
            inference_state["obj_id_to_idx"] = {7: 0}
            inference_state["obj_ids"] = [7]

        def add_new_points(self, *_args, **kwargs) -> None:
            self.added_points = kwargs

        def propagate_in_video_preflight(self, *_args, **kwargs) -> None:
            pass

        def _get_obj_num(self, inference_state) -> int:
            return len(inference_state["obj_ids"])

        def _run_single_frame_inference(self, *_args, **kwargs):
            self.run_single_frame_kwargs = kwargs
            return (
                {"object_score_logits": torch.tensor([[0.25]])},
                torch.ones(1, 1, 12, 20),
            )

        def _get_orig_video_res_output(self, inference_state, pred_masks):
            return None, pred_masks

    model = FakeVideoModel()
    predictor = object.__new__(VideoMemoryInference)
    predictor.model = model
    predictor.device = torch.device("cpu")
    predictor.image_size = 16
    reference = MemoryReference(
        image=Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8)),
        mask=np.ones((8, 10), dtype=bool),
        obj_id=7,
    )
    target = Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8))

    prediction = predictor.predict(
        target_image=target,
        references=[reference],
        target_point_coords=np.array([[10, 6]], dtype=np.float32),
        target_point_labels=np.array([1], dtype=np.int64),
        target_point_mode="memory",
    )

    assert model.added_points is None
    assert model.run_single_frame_kwargs["frame_idx"] == 1
    assert model.run_single_frame_kwargs["is_init_cond_frame"] is False
    assert model.run_single_frame_kwargs["mask_inputs"] is None
    assert model.run_single_frame_kwargs["objects_to_interact"] == [0]
    assert model.run_single_frame_kwargs["run_mem_encoder"] is False
    assert model.run_single_frame_kwargs["point_inputs"]["point_coords"].shape == (
        1,
        1,
        2,
    )
    assert prediction.frame_index == 1
    assert prediction.obj_ids == [7]

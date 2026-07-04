from __future__ import annotations

import numpy as np
import torch
from PIL import Image


def _tiny_image() -> Image.Image:
    return Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))


def _tiny_mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return mask


def test_video_memory_public_api_imports() -> None:
    from src import Sam3MemoryPredictor, Sam3MemoryReference, build_video_memory_model

    assert Sam3MemoryPredictor.__name__ == "Sam3MemoryPredictor"
    assert Sam3MemoryReference.__name__ == "Sam3MemoryReference"
    assert callable(build_video_memory_model)


def test_video_memory_modules_import_without_triton() -> None:
    import src.tracker_utils
    import src.video_tracking_multiplex
    import src.video_tracking_multiplex_demo

    assert hasattr(src.tracker_utils, "select_closest_cond_frames")
    assert hasattr(src.video_tracking_multiplex, "VideoTrackingDynamicMultiplex")
    assert hasattr(src.video_tracking_multiplex_demo, "Sam3VideoTrackingMultiplexDemo")


def test_video_checkpoint_remap_keeps_tracker_memory_and_backbone_keys() -> None:
    from src.video_checkpoint import filter_and_remap_video_state_dict

    checkpoint = {
        "model": {
            "tracker.model.maskmem_backbone.mask_downsampler.encoder.0.weight": torch.zeros(
                1
            ),
            "tracker.model.sam_mask_decoder.iou_token.weight": torch.zeros(1),
            "tracker.model.transformer.encoder.layers.0.self_attn_q_proj.weight": torch.zeros(
                1
            ),
            "tracker.model.interactive_sam_prompt_encoder.no_mask_embed.weight": torch.zeros(
                1
            ),
            "detector.backbone.vision_backbone.trunk.patch_embed.proj.weight": torch.zeros(
                1
            ),
            "detector.backbone.language_backbone.encoder.foo": torch.zeros(1),
        }
    }

    remapped, ignored = filter_and_remap_video_state_dict(checkpoint)

    assert (
        "maskmem_backbone.mask_downsampler.encoder.0.weight" in remapped
    )
    assert "sam_mask_decoder.iou_token.weight" in remapped
    assert "transformer.encoder.layers.0.self_attn_q_proj.weight" in remapped
    assert "interactive_sam_prompt_encoder.no_mask_embed.weight" in remapped
    assert "backbone.vision_backbone.trunk.patch_embed.proj.weight" in remapped
    assert ignored == ["detector.backbone.language_backbone.encoder.foo"]


def test_memory_references_preserve_order_for_same_object_id() -> None:
    from src.memory_predictor import Sam3MemoryPredictor, Sam3MemoryReference

    references = [
        Sam3MemoryReference(image=_tiny_image(), mask=_tiny_mask(), obj_id=3),
        Sam3MemoryReference(image=_tiny_image(), mask=_tiny_mask(), obj_id=3),
        Sam3MemoryReference(image=_tiny_image(), mask=_tiny_mask(), obj_id=9),
    ]

    prepared = Sam3MemoryPredictor.prepare_references(references)

    assert [item.frame_index for item in prepared] == [0, 1, 2]
    assert [item.reference.obj_id for item in prepared] == [3, 3, 9]


def test_preprocess_sequence_uses_target_size_for_mixed_image_sizes() -> None:
    from src.memory_predictor import Sam3MemoryPredictor

    predictor = object.__new__(Sam3MemoryPredictor)
    predictor.image_size = 16
    reference = Image.fromarray(np.zeros((8, 10, 3), dtype=np.uint8))
    target = Image.fromarray(np.zeros((12, 20, 3), dtype=np.uint8))

    batch, orig_hw, frame_hws = predictor._preprocess_image_sequence(
        [reference, target],
        output_image_index=1,
    )

    assert tuple(batch.shape) == (2, 3, 16, 16)
    assert orig_hw == (12, 20)
    assert frame_hws == [(8, 10), (12, 20)]


def test_mask_to_tensor_resizes_reference_mask_to_target_size() -> None:
    from src.memory_predictor import Sam3MemoryPredictor

    predictor = object.__new__(Sam3MemoryPredictor)
    mask = np.zeros((8, 10), dtype=bool)
    mask[2:6, 3:7] = True

    resized = predictor._mask_to_tensor(
        mask,
        source_hw=(8, 10),
        target_hw=(12, 20),
    )

    assert tuple(resized.shape) == (1, 12, 20)
    assert resized.dtype == torch.float32
    assert resized.sum() > mask.sum()


def test_memory_predictor_adds_target_points_after_reference_masks() -> None:
    from src.memory_predictor import Sam3MemoryPredictor, Sam3MemoryReference

    class FakeVideoModel:
        image_size = 16

        def __init__(self) -> None:
            self.added_points = None

        def init_state(self, **kwargs):
            return {"device": torch.device("cpu"), **kwargs}

        def add_new_masks(self, *args, **kwargs) -> None:
            pass

        def add_new_points(self, *args, **kwargs) -> None:
            self.added_points = kwargs

        def propagate_in_video_preflight(self, *args, **kwargs) -> None:
            pass

        def propagate_in_video(self, *args, **kwargs):
            yield (
                1,
                [7],
                None,
                torch.ones(1, 1, 12, 20),
                torch.tensor([[0.5]]),
            )

    model = FakeVideoModel()
    predictor = object.__new__(Sam3MemoryPredictor)
    predictor.model = model
    predictor.device = torch.device("cpu")
    predictor.image_size = 16
    predictor.load_report = None
    reference = Sam3MemoryReference(
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
    from src.memory_predictor import Sam3MemoryPredictor

    class FakeVideoModel:
        image_size = 16

        def __init__(self) -> None:
            self.added_masks = 0
            self.added_points = None

        def init_state(self, **kwargs):
            return {"device": torch.device("cpu"), **kwargs}

        def add_new_masks(self, *args, **kwargs) -> None:
            self.added_masks += 1

        def add_new_points(self, *args, **kwargs) -> None:
            self.added_points = kwargs

        def propagate_in_video_preflight(self, *args, **kwargs) -> None:
            pass

        def propagate_in_video(self, *args, **kwargs):
            yield (
                0,
                [5],
                None,
                torch.ones(1, 1, 12, 20),
                torch.tensor([[0.75]]),
            )

    model = FakeVideoModel()
    predictor = object.__new__(Sam3MemoryPredictor)
    predictor.model = model
    predictor.device = torch.device("cpu")
    predictor.image_size = 16
    predictor.load_report = None
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

import numpy as np
import pytest
import torch
from PIL import Image
from src.predict.ground import GroundPredictor


class FakeGroundModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_calls = 0
        self.image_grad_enabled = []
        self.prompt_calls = []
        self.decode_batches = []

    def encode_image(self, _image):
        self.image_calls += 1
        self.image_grad_enabled.append(torch.is_grad_enabled())
        features = torch.zeros(1, 2, 2, 2)
        features[:, 0] = 1
        return {"backbone_fpn": (features,)}

    def encode_box_prompts(self, image, boxes, labels, box_mask):
        self.prompt_calls.append((image, boxes, labels, box_mask))
        batch = boxes.shape[1]
        return {
            "features": torch.zeros(3, batch, 2),
            "mask": torch.zeros(batch, 3, dtype=torch.bool),
        }

    def decode(self, _image, prompt):
        batch = prompt["features"].shape[1]
        self.decode_batches.append(batch)
        return {
            "pred_logits": torch.full((batch, 1, 1), 4.0),
            "pred_boxes": torch.tensor([0.5, 0.5, 0.5, 0.5]).repeat(batch, 1, 1),
            "pred_masks": torch.full((batch, 1, 2, 2), 2.0),
            "raw": {"unused": torch.ones(10)},
        }


class FakeSamMaskGroundModel(FakeGroundModel):
    def __init__(self):
        super().__init__()
        self.refine_images = []
        self.mask_prompts = []
        self.sam_prompts = []
        self.stability_values = []

    @property
    def mask_input_size(self):
        return (2, 2)

    def encode_image(self, image):
        out = super().encode_image(image)
        out["mask_image_embed"] = torch.zeros(1, 2, 2, 2)
        out["mask_high_res"] = ()
        return out

    def encode_mask_prompts(self, points, masks=None):
        self.mask_prompts.append(masks)
        self.sam_prompts.append(points)
        return points

    def decode_sam_mask(self, image, prompt):
        self.refine_images.append(image)
        batch = prompt[0].shape[0]
        return (
            torch.full((batch, 1, 2, 2), 2.0),
            torch.full((batch, 1), 0.95),
        )

    def mask_stability(self, masks):
        value = self.stability_values.pop(0) if self.stability_values else 0.9
        return torch.full(masks.shape[:2], value)

    def decode_point(self, image, prompt):
        self.refine_images.append(image)
        batch = prompt[0].shape[0]
        masks = torch.full((batch, 3, 4, 4), -2.0)
        masks[:, 1, 1:3, 1:3] = 2.0
        scores = torch.tensor([[0.1, 0.9, 0.2]]).repeat(batch, 1)
        return masks, scores


class FakeDisconnectedSamMaskModel(FakeSamMaskGroundModel):
    def decode_sam_mask(self, image, prompt):
        self.refine_images.append(image)
        batch = prompt[0].shape[0]
        masks = torch.full((batch, 1, 4, 4), -2.0)
        masks[:, :, 0, :2] = 2.0
        masks[:, :, 3, 3] = 2.0
        return masks, torch.full((batch, 1), 0.95)


def test_encode_reference_groups_boxes_by_class_and_encodes_image_once():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")

    reference = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[0, 0, 4, 4], [4, 4, 8, 8], [1, 1, 3, 3]],
        [2, 1, 2],
    )

    assert model.image_calls == 1
    assert reference["prompt_classes"].tolist() == [1, 2]
    assert reference["feature_classes"].tolist() == [2, 1, 2]
    _image, boxes, _labels, box_mask = model.prompt_calls[0]
    assert boxes.shape == (2, 2, 4)
    assert box_mask.tolist() == [[False, True], [False, False]]


def test_encode_uses_inference_mode():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")

    predictor.encode(Image.new("RGB", (8, 8)))

    assert model.image_grad_enabled == [False]


def test_predict_encodes_target_once_and_decodes_prompt_batches():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu", prompt_batch_size=2)
    boxes = [[index, 0, index + 1, 2] for index in range(5)]
    reference = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        boxes,
        [0, 1, 2, 3, 4],
    )

    objects = predictor.predict(Image.new("RGB", (8, 8)), [reference])

    assert model.image_calls == 2
    assert model.decode_batches == [2, 2, 1]
    assert [item["class_id"] for item in objects] == [0, 1, 2, 3, 4]
    assert all("raw" not in item for item in objects)
    assert all(isinstance(item["roi"], np.ndarray) for item in objects)
    assert all("mask" not in item for item in objects)


def test_predict_embed_reuses_encoded_target():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    reference = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[0, 0, 4, 4]],
        [1],
    )
    target = predictor.encode(Image.new("RGB", (8, 8)))

    first = predictor.predict_embed(target, [reference])
    second = predictor.predict_embed(target, [reference])

    assert model.image_calls == 2
    assert len(first) == len(second) == 1
    assert first[0]["class_id"] == second[0]["class_id"] == 1


def test_prompt_session_accumulates_feedback_without_reencoding_image():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))

    first = predictor.add_prompt(state, [0, 0, 4, 4])
    second = predictor.add_prompt(state, [4, 4, 8, 8])
    third = predictor.add_prompt(state, [0, 0, 8, 8], positive=False)

    assert model.image_calls == 1
    assert [call[2][:, 0].tolist() for call in model.prompt_calls] == [
        [1],
        [1, 1],
        [1, 1, 0],
    ]
    assert len(first) == len(second) == 1
    assert third == []
    assert state["box_labels"] == [1, 1, 0]


def test_prompt_session_adds_sam_mask_to_every_candidate_without_a_count_limit():
    model = FakeSamMaskGroundModel()
    predictor = GroundPredictor(
        model,
        device="cpu",
        top_k=None,
        use_sam_masks=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))

    objects = predictor.add_prompt(state, [0, 0, 4, 4])

    assert len(objects) == 1
    assert objects[0]["box"] == (0, 0, 8, 8)
    assert objects[0]["metrics"]["mask_score"] == pytest.approx(0.95)
    assert objects[0]["metrics"]["stability_score"] == pytest.approx(0.9)
    assert model.refine_images[0]["mask_image_embed"].shape == (1, 2, 2, 2)


def test_refine_objects_feeds_current_logits_back_as_mask_prompts():
    model = FakeSamMaskGroundModel()
    predictor = GroundPredictor(
        model,
        device="cpu",
        use_sam_masks=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))
    objects = predictor.add_prompt(state, [0, 0, 4, 4])
    objects[0]["logit"] = np.array([[1.0, -1.0], [-2.0, 2.0]])
    model.stability_values.append(0.900001)

    refined = predictor.refine_objects(state, objects)

    assert torch.equal(
        model.mask_prompts[-1],
        torch.tensor([[[[1.0, -1.0], [-2.0, 2.0]]]]),
    )
    assert model.sam_prompts[-1][1].tolist() == [[-1]]
    assert refined[0]["metrics"]["refine_score"] == pytest.approx(0.95)
    assert refined[0]["metrics"]["stability_score"] == pytest.approx(0.900001)

    calls = len(model.mask_prompts)
    repeated = predictor.refine_objects(state, refined)

    assert len(repeated) == 1
    assert len(model.mask_prompts) == calls + 1


@pytest.mark.parametrize("stability", [0.8, 0.9])
def test_refine_objects_keeps_previous_mask_when_stability_does_not_improve(
    stability,
):
    model = FakeSamMaskGroundModel()
    predictor = GroundPredictor(
        model,
        device="cpu",
        use_sam_masks=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))
    objects = predictor.add_prompt(state, [0, 0, 4, 4])
    objects[0]["logit"] = np.array([[1.0, -1.0], [-2.0, 2.0]])
    previous = objects[0]["logit"].copy()
    model.stability_values.append(stability)

    refined = predictor.refine_objects(state, objects)

    assert np.array_equal(refined[0]["logit"], previous)
    assert refined[0]["metrics"]["stability_score"] == pytest.approx(0.9)
    assert "refine_score" not in refined[0]["metrics"]


def test_prompt_session_can_keep_only_largest_connected_mask_component():
    predictor = GroundPredictor(
        FakeDisconnectedSamMaskModel(),
        device="cpu",
        use_sam_masks=True,
        largest_component=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))

    objects = predictor.add_prompt(state, [0, 0, 4, 4])

    assert len(objects) == 1
    assert objects[0]["box"][2] < 8
    assert objects[0]["box"][3] < 8


def test_point_session_uses_sam_mask_as_positive_and_negative_examples():
    model = FakeSamMaskGroundModel()
    predictor = GroundPredictor(
        model,
        device="cpu",
        use_sam_masks=True,
        largest_component=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))

    positive = predictor.add_point(state, [4, 4])
    negative = predictor.add_point(state, [1, 1], positive=False)

    assert len(positive) == 1
    assert negative == []
    assert model.image_calls == 1
    assert [point.tolist() for point in state["points"]] == [[4, 4], [1, 1]]
    assert state["box_labels"] == [1, 0]
    assert all(box.shape == (4,) for box in state["boxes"])


def test_point_session_moves_selected_point_and_recomputes_its_object_box():
    predictor = GroundPredictor(
        FakeSamMaskGroundModel(),
        device="cpu",
        use_sam_masks=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_point(state, [2, 2])

    objects = predictor.update_point(state, 0, [6, 5])

    assert len(objects) == 1
    assert state["points"][0].tolist() == [6, 5]


def test_point_session_requires_sam_masks_and_positive_point_first():
    plain = GroundPredictor(FakeGroundModel(), device="cpu")
    plain_state = plain.start(Image.new("RGB", (8, 8)))
    with pytest.raises(RuntimeError, match="SAM masks"):
        plain.add_point(plain_state, [2, 2])

    refined = GroundPredictor(
        FakeSamMaskGroundModel(),
        device="cpu",
        use_sam_masks=True,
    )
    refined_state = refined.start(Image.new("RGB", (8, 8)))
    with pytest.raises(ValueError, match="positive point"):
        refined.add_point(refined_state, [2, 2], positive=False)


def test_prompt_session_requires_positive_box_first():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))

    with pytest.raises(ValueError, match="positive"):
        predictor.add_prompt(state, [0, 0, 4, 4], positive=False)

    assert state["boxes"] == []
    assert state["box_labels"] == []


def test_prompt_session_removes_last_feedback_and_repredicts():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_prompt(state, [0, 0, 4, 4])
    predictor.add_prompt(state, [4, 4, 8, 8])

    objects = predictor.remove_prompt(state)
    empty = predictor.remove_prompt(state)

    assert len(objects) == 1
    assert empty == []
    assert state["boxes"] == []
    assert state["box_labels"] == []


def test_prompt_session_updates_and_removes_selected_prompt():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_prompt(state, [0, 0, 4, 4])
    predictor.add_prompt(state, [4, 4, 8, 8])

    updated = predictor.update_prompt(state, 0, [1, 1, 5, 5])
    removed = predictor.remove_prompt_at(state, 0)

    assert len(updated) == len(removed) == 1
    assert state["boxes"][0].tolist() == [4, 4, 8, 8]
    assert state["box_labels"] == [1]


@pytest.mark.parametrize(
    "operation",
    [
        "add_prompt",
        "add_point",
        "remove_prompt",
        "update_prompt",
        "update_point",
        "remove_prompt_at",
        "remove_prompts_at",
    ],
)
def test_prompt_mutation_rolls_back_when_prediction_fails(monkeypatch, operation):
    predictor = GroundPredictor(
        FakeSamMaskGroundModel(),
        device="cpu",
        use_sam_masks=True,
    )
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_prompt(state, [0, 0, 4, 4])
    predictor.add_prompt(state, [4, 4, 8, 8])
    previous_boxes = [box.copy() for box in state["boxes"]]
    previous_labels = list(state["box_labels"])
    previous_points = list(state["points"])

    def fail(_state):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(predictor, "predict_prompt", fail)

    with pytest.raises(RuntimeError, match="decode failed"):
        if operation == "add_prompt":
            predictor.add_prompt(state, [1, 1, 7, 7])
        elif operation == "add_point":
            predictor.add_point(state, [2, 2])
        elif operation == "remove_prompt":
            predictor.remove_prompt(state)
        elif operation == "update_prompt":
            predictor.update_prompt(state, 0, [1, 1, 5, 5])
        elif operation == "update_point":
            predictor.update_point(state, 0, [2, 2])
        elif operation == "remove_prompt_at":
            predictor.remove_prompt_at(state, 0)
        else:
            predictor.remove_prompts_at(state, [0, 1])

    assert state["box_labels"] == previous_labels
    assert state["points"] == previous_points
    assert all(
        np.array_equal(box, previous)
        for box, previous in zip(state["boxes"], previous_boxes, strict=True)
    )


def test_prompt_session_removes_multiple_selected_points_once():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_prompt(state, [0, 0, 2, 2])
    predictor.add_prompt(state, [2, 2, 4, 4])
    predictor.add_prompt(state, [4, 4, 6, 6], positive=False)

    objects = predictor.remove_prompts_at(state, [0, 2])

    assert len(objects) == 1
    assert len(state["boxes"]) == 1
    assert state["box_labels"] == [1]


def test_prompt_session_keeps_state_when_batch_delete_removes_all_positives():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_prompt(state, [0, 0, 2, 2])
    predictor.add_prompt(state, [2, 2, 4, 4], positive=False)

    with pytest.raises(ValueError, match="positive point"):
        predictor.remove_prompts_at(state, [0])

    assert len(state["boxes"]) == 2
    assert state["box_labels"] == [1, 0]


@pytest.mark.parametrize("index", [-1, 1, True, 0.5])
def test_prompt_session_rejects_invalid_selected_prompt(index):
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    state = predictor.start(Image.new("RGB", (8, 8)))
    predictor.add_prompt(state, [0, 0, 4, 4])

    error = TypeError if isinstance(index, (bool, float)) else IndexError
    with pytest.raises(error):
        predictor.update_prompt(state, index, [1, 1, 5, 5])
    with pytest.raises(error):
        predictor.remove_prompt_at(state, index)


def test_predict_merges_same_class_features_across_references():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    first = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[0, 0, 4, 4]],
        [3],
    )
    second = predictor.encode_reference(
        Image.new("RGB", (8, 8)),
        [[4, 4, 8, 8]],
        [3],
    )

    objects = predictor.predict(Image.new("RGB", (8, 8)), [first, second])

    assert len(objects) == 1
    assert objects[0]["class_id"] == 3


def test_predict_requires_non_empty_reference_list():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    image = Image.new("RGB", (8, 8))

    with pytest.raises(TypeError, match="list"):
        predictor.predict(image, {})
    with pytest.raises(ValueError, match="empty"):
        predictor.predict(image, [])
    assert model.image_calls == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 0},
        {"score_thr": -0.1},
        {"score_thr": 1.1},
        {"nms_thr": -0.1},
        {"nms_thr": 1.1},
        {"sim_thr": -1.1},
        {"sim_thr": 1.1},
        {"negative_margin": -0.1},
        {"negative_margin": 2.1},
        {"prompt_batch_size": 0},
        {"prompt_batch_size": -1},
        {"prompt_batch_size": 1.5},
        {"prompt_batch_size": True},
    ],
)
def test_predictor_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        GroundPredictor(FakeGroundModel(), device="cpu", **kwargs)


def test_predictor_removes_old_reference_options():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")

    assert not hasattr(predictor, "encode_ref")
    with pytest.raises(TypeError):
        GroundPredictor(FakeGroundModel(), device="cpu", score_thresh=0.5)

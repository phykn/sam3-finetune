import inspect

import numpy as np
import torch


class FakeBackbone:
    def __init__(self) -> None:
        self.forwarded_images = []
        self.forwarded_text = []

    def forward_image(self, image: torch.Tensor) -> dict[str, object]:
        self.forwarded_images.append(image.detach().clone())
        batch = image.shape[0]
        features = torch.zeros(batch, 4, 2, 2, dtype=torch.float32)
        pos = torch.zeros_like(features)
        return {
            "vision_features": features,
            "vision_mask": None,
            "vision_pos_enc": [pos],
            "backbone_fpn": [features],
        }

    def forward_text(
        self, captions, input_boxes=None, additional_text=None, device="cpu"
    ):
        self.forwarded_text.append(
            {
                "captions": list(captions),
                "input_boxes": input_boxes,
                "additional_text": additional_text,
                "device": str(device),
            }
        )
        count = len(captions)
        return {
            "language_features": torch.ones(3, count, 4, dtype=torch.float32),
            "language_mask": torch.zeros(count, 3, dtype=torch.bool),
            "language_embeds": torch.ones(3, count, 4, dtype=torch.float32),
        }


class FakeGroundingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()
        self.forward_calls = []

    def forward_grounding(
        self,
        *,
        backbone_out,
        find_input,
        geometric_prompt,
        visual_prompt_embed=None,
        visual_prompt_mask=None,
        encode_text=True,
    ):
        assert visual_prompt_embed is None
        assert visual_prompt_mask is None
        assert encode_text is True
        assert find_input.text_ids.tolist() == [0]
        assert "language_features" in backbone_out
        assert "language_mask" in backbone_out
        self.forward_calls.append(
            {
                "box_embeddings": geometric_prompt.box_embeddings.detach().clone(),
                "box_labels": geometric_prompt.box_labels.detach().clone(),
                "text_ids": find_input.text_ids.detach().clone(),
                "encode_text": encode_text,
                "has_language": "language_features" in backbone_out,
            }
        )
        return {
            "pred_logits": torch.tensor([[[4.0]]], dtype=torch.float32),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5]]], dtype=torch.float32),
            "pred_masks": torch.ones(1, 1, 4, 4, dtype=torch.float32),
        }

    def get_dummy_prompt(self, num_prompts: int = 1):
        from src.model.grounding.prompt import Prompt

        return Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4),
            box_mask=torch.zeros(num_prompts, 0, dtype=torch.bool),
            box_labels=torch.zeros(0, num_prompts, dtype=torch.bool),
        )


def _image(width: int = 8, height: int = 8) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _mask(x0: int, y0: int, x1: int, y1: int, *, width: int = 8, height: int = 8):
    mask = np.zeros((height, width), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_visual_prompt_encoder_converts_mask_to_normalized_box_prompt():
    from src.predict.visual_prompt.encoder import VisualPromptEncoder
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    encoder = VisualPromptEncoder(model, device="cpu")

    prepared = encoder.prepare(
        [
            VisualExemplar(
                image=_image(width=10, height=20),
                mask=_mask(2, 4, 8, 14, width=10, height=20),
                concept_id=3,
            )
        ]
    )

    assert len(prepared.concepts) == 1
    concept = prepared.concepts[0]
    assert concept.concept_id == 3
    torch.testing.assert_close(
        concept.boxes_cxcywh.cpu(),
        torch.tensor([[[0.5, 0.45, 0.6, 0.5]]], dtype=torch.float32),
    )
    assert concept.box_labels.tolist() == [[True]]
    assert model.backbone.forwarded_text[0]["captions"] == ["visual"]
    assert model.backbone.forwarded_text[0]["input_boxes"] is None
    assert concept.language_features.shape == (3, 1, 4)
    assert concept.language_mask.shape == (1, 3)


def test_visual_prompt_predictor_uses_vlm_visual_text_and_box_prompt():
    from src.predict.visual_prompt.predictor import VisualPromptPredictor
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    predictor = VisualPromptPredictor(model, device="cpu", image_size=8)

    predictions = predictor.predict(
        target_image=_image(),
        exemplars=[
            VisualExemplar(
                image=_image(),
                mask=_mask(1, 1, 5, 5),
                concept_id=2,
            )
        ],
        confidence_threshold=0.0,
    )

    assert len(predictions) == 1
    assert predictions[0].concept_id == 2
    assert predictions[0].masks.shape == (1, 8, 8)
    assert model.backbone.forwarded_text[0]["captions"] == ["visual"]
    assert len(model.forward_calls) == 1
    call = model.forward_calls[0]
    assert call["has_language"]
    assert call["encode_text"] is True
    torch.testing.assert_close(
        call["box_embeddings"],
        torch.tensor([[[0.375, 0.375, 0.5, 0.5]]], dtype=torch.float32),
    )
    assert call["box_labels"].tolist() == [[True]]


def test_visual_prompt_path_does_not_use_context_prototype_or_geometry_mask_encoder():
    import src.predict.visual_prompt.encoder as encoder_module
    import src.predict.visual_prompt.predictor as predictor_module

    source = inspect.getsource(encoder_module) + inspect.getsource(predictor_module)

    assert "build_context_prototype" not in source
    assert "_masked_feature_mean" not in source
    assert "ContextPrototype" not in source
    assert "mask_encoder" not in source
    assert "mask_prompt_encoder" not in source
    assert "visual_prompt_embed" not in source
    assert "encode_text=False" not in source


def test_visual_prompt_predictor_runs_each_concept_with_separate_vlm_features():
    from src.predict.visual_prompt.predictor import VisualPromptPredictor
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    predictor = VisualPromptPredictor(model, device="cpu", image_size=8)

    predictions = predictor.predict(
        target_image=_image(),
        exemplars=[
            VisualExemplar(image=_image(), mask=_mask(0, 0, 2, 2), concept_id=0),
            VisualExemplar(image=_image(), mask=_mask(0, 0, 4, 4), concept_id=1),
        ],
        confidence_threshold=0.0,
    )

    assert [prediction.concept_id for prediction in predictions] == [0, 1]
    assert len(model.backbone.forwarded_text) == 2
    assert len(model.forward_calls) == 2
    first = model.forward_calls[0]["box_embeddings"]
    second = model.forward_calls[1]["box_embeddings"]
    assert not torch.equal(first, second)


def test_visual_prompt_encoder_groups_multiple_exemplars_as_multiple_boxes():
    from src.predict.visual_prompt.encoder import VisualPromptEncoder
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    encoder = VisualPromptEncoder(model, device="cpu")

    prepared = encoder.prepare(
        [
            VisualExemplar(image=_image(), mask=_mask(0, 0, 2, 2), concept_id=4),
            VisualExemplar(image=_image(), mask=_mask(0, 0, 3, 3), concept_id=4),
        ]
    )

    assert len(prepared.concepts) == 1
    concept = prepared.concepts[0]
    assert concept.concept_id == 4
    assert len(concept.exemplars) == 2
    assert concept.boxes_cxcywh.shape == (2, 1, 4)
    assert concept.box_labels.shape == (2, 1)
    assert model.backbone.forwarded_text == [
        {
            "captions": ["visual"],
            "input_boxes": None,
            "additional_text": None,
            "device": "cpu",
        }
    ]


def test_visual_prompt_package_exports_user_facing_api():
    import src.predict.visual_prompt as visual_prompt
    from src.predict.visual_prompt.encoder import VisualPromptEncoder
    from src.predict.visual_prompt.predictor import VisualPromptPredictor
    from src.predict.visual_prompt.types import VisualExemplar, VisualPromptPrediction

    assert visual_prompt.VisualPromptEncoder is VisualPromptEncoder
    assert visual_prompt.VisualPromptPredictor is VisualPromptPredictor
    assert visual_prompt.VisualExemplar is VisualExemplar
    assert visual_prompt.VisualPromptPrediction is VisualPromptPrediction
    assert not hasattr(visual_prompt, "__all__")

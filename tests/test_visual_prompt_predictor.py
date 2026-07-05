import inspect

import numpy as np
import torch


class FakeBackbone:
    def __init__(self) -> None:
        self.forwarded_images = []

    def forward_image(self, image: torch.Tensor) -> dict[str, object]:
        self.forwarded_images.append(image.detach().clone())
        batch = image.shape[0]
        value = float(len(self.forwarded_images))
        features = torch.full((batch, 4, 2, 2), value, dtype=torch.float32)
        pos = torch.zeros_like(features)
        return {
            "vision_features": features,
            "vision_mask": None,
            "vision_pos_enc": [pos],
            "backbone_fpn": [features],
        }


class FakeGeometryEncoder:
    def __init__(self) -> None:
        self.mask_encoder = object()
        self.calls = []

    def __call__(self, geo_prompt, img_feats, img_sizes, img_pos_embeds=None):
        assert geo_prompt.mask_embeddings is not None
        assert img_feats[-1].shape[-1] == 4
        self.calls.append((geo_prompt, img_feats, img_sizes, img_pos_embeds))
        token_value = geo_prompt.mask_embeddings.float().sum()
        token = torch.full((1, 1, 4), float(token_value), dtype=torch.float32)
        mask = torch.zeros(1, 1, dtype=torch.bool)
        return token, mask


class FakeGroundingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = FakeBackbone()
        self.geometry_encoder = FakeGeometryEncoder()
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
        assert visual_prompt_embed is not None
        assert visual_prompt_mask is not None
        assert encode_text is False
        assert find_input.text_ids.numel() == 0
        self.forward_calls.append(
            {
                "visual_prompt_embed": visual_prompt_embed.detach().clone(),
                "visual_prompt_mask": visual_prompt_mask.detach().clone(),
                "encode_text": encode_text,
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
        )


def _image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    mask = np.zeros((8, 8), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def test_visual_prompt_encoder_creates_embed_from_image_mask_exemplar():
    from src.predict.visual_prompt.encoder import VisualPromptEncoder
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    encoder = VisualPromptEncoder(model, device="cpu", image_size=8)

    prepared = encoder.prepare(
        [
            VisualExemplar(
                image=_image(),
                mask=_mask(1, 1, 4, 4),
                concept_id=3,
            )
        ]
    )

    assert len(prepared.concepts) == 1
    concept = prepared.concepts[0]
    assert concept.concept_id == 3
    assert concept.visual_prompt_embed.shape == (1, 1, 4)
    assert concept.visual_prompt_mask.shape == (1, 1)
    assert model.geometry_encoder.calls


def test_visual_prompt_predictor_passes_visual_prompt_to_grounding_without_text():
    from src.predict.visual_prompt.predictor import VisualPromptPredictor
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    predictor = VisualPromptPredictor(model, device="cpu", image_size=8)

    predictions = predictor.predict(
        target_image=_image(),
        exemplars=[
            VisualExemplar(
                image=_image(),
                mask=_mask(1, 1, 4, 4),
                concept_id=2,
            )
        ],
        confidence_threshold=0.0,
    )

    assert len(predictions) == 1
    assert predictions[0].concept_id == 2
    assert predictions[0].masks.shape == (1, 8, 8)
    assert len(model.forward_calls) == 1
    call = model.forward_calls[0]
    assert call["visual_prompt_embed"].shape == (1, 1, 4)
    assert call["visual_prompt_mask"].shape == (1, 1)
    assert call["encode_text"] is False


def test_visual_prompt_path_does_not_use_context_prototype_average():
    import src.predict.visual_prompt.encoder as encoder_module
    import src.predict.visual_prompt.predictor as predictor_module

    source = inspect.getsource(encoder_module) + inspect.getsource(predictor_module)

    assert "build_context_prototype" not in source
    assert "_masked_feature_mean" not in source
    assert "ContextPrototype" not in source


def test_visual_prompt_predictor_runs_each_concept_with_separate_prompt_tokens():
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
    assert len(model.forward_calls) == 2
    first = model.forward_calls[0]["visual_prompt_embed"]
    second = model.forward_calls[1]["visual_prompt_embed"]
    assert not torch.equal(first, second)


def test_visual_prompt_encoder_groups_multiple_exemplars_for_same_concept():
    from src.predict.visual_prompt.encoder import VisualPromptEncoder
    from src.predict.visual_prompt.types import VisualExemplar

    model = FakeGroundingModel()
    encoder = VisualPromptEncoder(model, device="cpu", image_size=8)

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
    assert concept.visual_prompt_embed.shape == (2, 1, 4)
    assert concept.visual_prompt_mask.shape == (1, 2)


def test_grounding_forward_accepts_visual_prompt_embed_and_mask():
    from src.model.grounding.model import GroundingImageModel
    from src.model.grounding.prompt import Prompt
    from src.model.structures import FindStage

    class MinimalGrounding(GroundingImageModel):
        def __init__(self) -> None:
            torch.nn.Module.__init__(self)
            self.seen = None

        def _select_image_features(self, backbone_out, img_ids):
            feature = torch.zeros(1, 1, 4)
            return [feature], [feature], [(1, 1)]

        def _encode_prompt(self, **kwargs):
            self.seen = kwargs
            return torch.zeros(1, 1, 4), torch.zeros(1, 1, dtype=torch.bool)

        def _run_encoder(self, **kwargs):
            return {
                "encoder_hidden_states": torch.zeros(1, 1, 4),
                "pos_embed": torch.zeros(1, 1, 4),
                "padding_mask": torch.zeros(1, 1, dtype=torch.bool),
                "level_start_index": torch.zeros(1, dtype=torch.long),
                "spatial_shapes": torch.ones(1, 2, dtype=torch.long),
                "valid_ratios": torch.ones(1, 1, 2),
                "vis_feat_sizes": [(1, 1)],
            }

        def _run_decoder(self, **kwargs):
            return kwargs["out"], torch.zeros(1, 1, 1, 4)

        def _run_segmentation_heads(self, **kwargs):
            return None

    model = MinimalGrounding()
    visual_prompt_embed = torch.ones(2, 1, 4)
    visual_prompt_mask = torch.zeros(1, 2, dtype=torch.bool)

    model.forward_grounding(
        backbone_out={},
        find_input=FindStage(
            img_ids=torch.tensor([0]),
            text_ids=torch.empty(0, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        ),
        geometric_prompt=Prompt(
            box_embeddings=torch.zeros(0, 1, 4),
            box_mask=torch.zeros(1, 0, dtype=torch.bool),
        ),
        visual_prompt_embed=visual_prompt_embed,
        visual_prompt_mask=visual_prompt_mask,
        encode_text=False,
    )

    assert model.seen["visual_prompt_embed"] is visual_prompt_embed
    assert model.seen["visual_prompt_mask"] is visual_prompt_mask
    assert model.seen["encode_text"] is False


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

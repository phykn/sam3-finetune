import torch
from torch import nn

from src.ml.components.transformer.decoder import TransformerDecoder
from src.ml.components.video.tracker.frame.features import get_image_feature
from src.ml.components.video.tracker.memory.context import select_closest_cond_frames


class PassDecoderLayer(nn.Module):
    def forward(self, tgt, presence_token=None, **kwargs):
        return tgt, presence_token


class ConstantHead(nn.Module):
    def forward(self, x):
        return torch.full((*x.shape[:-1], 1), 99.0, device=x.device)


def test_presence_logits_are_clamped():
    decoder = TransformerDecoder(
        d_model=4,
        frozen=False,
        interaction_layer=None,
        layer=PassDecoderLayer(),
        num_layers=1,
        num_queries=1,
        return_intermediate=True,
        box_refine=True,
        presence_token=True,
        clamp_presence_logit_max_val=10.0,
    )
    decoder.eval()
    decoder.presence_token_head = ConstantHead()
    decoder.presence_token_out_norm = nn.Identity()

    _, _, presence_logits, _ = decoder(
        tgt=torch.zeros(1, 1, 4),
        memory=torch.zeros(1, 1, 4),
        memory_key_padding_mask=None,
        pos=torch.zeros(1, 1, 4),
        reference_boxes=torch.full((1, 1, 4), 0.5),
        level_start_index=torch.tensor([0]),
        spatial_shapes=torch.tensor([[1, 1]]),
        valid_ratios=torch.ones(1, 1, 2),
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_text=None,
        text_attention_mask=None,
        is_instance_prompt=False,
    )

    assert presence_logits.max().item() == 10.0


class FakeFeatureModel:
    def __init__(self):
        self.calls = 0

    def forward_image(self, image, **kwargs):
        self.calls += 1
        return {"new": image}

    def _prepare_backbone_features(self, backbone_out):
        return backbone_out


def test_image_feature_cache_miss_preserves_existing_entries():
    model = FakeFeatureModel()
    inference_state = {
        "cached_features": {0: ("image0", "features0")},
        "images": [torch.zeros(3, 2, 2), torch.ones(3, 2, 2)],
        "device": torch.device("cpu"),
    }

    image, features = get_image_feature(model, inference_state, frame_idx=1, batch_size=1)

    assert model.calls == 1
    assert image.shape == (1, 3, 2, 2)
    assert features["new"].tensors.shape == (1, 3, 2, 2)
    assert set(inference_state["cached_features"]) == {0, 1}
    assert inference_state["cached_features"][0] == ("image0", "features0")


def test_select_closest_cond_frames_respects_limit_with_first_anchor():
    cond_outputs = {
        0: "first",
        8: "before",
        12: "after",
        30: "far",
    }

    selected, unselected = select_closest_cond_frames(
        frame_idx=10,
        cond_frame_outputs=cond_outputs,
        max_cond_frame_num=2,
        keep_first_cond_frame=True,
    )

    assert len(selected) == 2
    assert 0 in selected
    assert set(selected) | set(unselected) == set(cond_outputs)

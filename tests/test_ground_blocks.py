from types import SimpleNamespace

import torch
from src.ml.blocks.ground_dec import GroundDec
from src.ops.tensor import invert_sigmoid


class FixedScorer:
    def __call__(self, hs, prompt, prompt_mask):
        return torch.full((*hs.shape[:-1], 1), 2.0)


class ZeroBox:
    dac = False

    def bbox_embed(self, hs):
        return torch.zeros(*hs.shape[:-1], 4)


def test_ground_decoder_applies_presence_to_scores():
    dec = SimpleNamespace(
        training=False,
        scorer=FixedScorer(),
        transformer=SimpleNamespace(decoder=ZeroBox()),
    )
    out = {}
    hs = torch.zeros(1, 1, 2, 4)
    refs = torch.full((1, 1, 2, 4), 0.5)
    prompt = torch.zeros(1, 1, 4)
    prompt_mask = torch.zeros(1, 1, dtype=torch.bool)
    presence = torch.zeros(1, 1, 1)

    GroundDec.score_and_box(dec, out, hs, refs, prompt, prompt_mask, presence)

    expected = invert_sigmoid(torch.sigmoid(torch.tensor(2.0)) * 0.5)
    torch.testing.assert_close(out["pred_logits"], expected.expand(1, 2, 1))

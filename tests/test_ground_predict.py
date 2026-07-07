import numpy as np
import torch
from PIL import Image
from src.predict.ground import GroundPredictor


class FakeGroundModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.images = []
        self.prompts = []
        self.decodes = []

    def encode_image(self, image):
        self.images.append(tuple(image.shape))
        return {
            "image": image,
            "backbone_fpn": (torch.ones(1, 2, 2, 2),),
        }

    def encode_prompt(self, image, **prompt):
        self.prompts.append(prompt)
        return {
            "features": torch.zeros(1, 1, 2),
            "mask": torch.zeros(1, 1, dtype=torch.bool),
            "prompt": prompt,
        }

    def decode(self, image, prompt):
        self.decodes.append((image, prompt))
        return {
            "pred_logits": torch.tensor([[[2.0], [-2.0]]]),
            "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.1, 0.1]]]),
            "pred_masks": torch.tensor(
                [
                    [
                        [[2.0, 2.0], [2.0, 2.0]],
                        [[-2.0, -2.0], [-2.0, -2.0]],
                    ]
                ]
            ),
            "raw": {},
        }


class FakeRefModel(FakeGroundModel):
    def encode_image(self, image):
        self.images.append(tuple(image.shape))
        fpn = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]]]])
        return {
            "image": image,
            "backbone_fpn": (fpn,),
        }

    def decode(self, image, prompt):
        self.decodes.append((image, prompt))
        return {
            "pred_logits": torch.tensor([[[4.0], [1.0]]]),
            "pred_boxes": torch.tensor(
                [[[0.75, 0.75, 0.4, 0.4], [0.25, 0.25, 0.4, 0.4]]]
            ),
            "pred_masks": torch.tensor(
                [
                    [
                        [[-2.0, -2.0], [-2.0, 2.0]],
                        [[2.0, -2.0], [-2.0, -2.0]],
                    ]
                ]
            ),
            "raw": {},
        }


def test_ground_predictor_encodes_mask_reference_as_visual_prompt():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 1:5] = True

    ref = predictor.encode_ref(Image.new("RGB", (8, 8)), mask=mask, name="frog")

    prompt = model.prompts[0]
    assert ref["name"] == "frog"
    assert prompt["boxes"] is None
    assert prompt["points"] is None
    assert prompt["masks"].shape == (1, 1, 1, 8, 8)
    assert ref["feature"].shape == (1, 2)


def test_ground_predictor_keeps_multiple_mask_reference_features():
    model = FakeGroundModel()
    predictor = GroundPredictor(model, device="cpu")
    masks = np.zeros((2, 8, 8), dtype=bool)
    masks[0, 2:6, 1:5] = True
    masks[1, :3, :3] = True

    ref = predictor.encode_ref(Image.new("RGB", (8, 8)), mask=masks, name="flower")

    prompt = model.prompts[0]
    assert prompt["masks"].shape == (2, 1, 1, 8, 8)
    assert ref["feature"].shape == (2, 2)


def test_ground_predictor_predicts_each_reference_separately():
    model = FakeGroundModel()
    predictor = GroundPredictor(
        model,
        device="cpu",
        score_thresh=0.5,
    )
    ref_a = predictor.encode_ref(
        Image.new("RGB", (8, 8)),
        box=[1, 1, 6, 6],
        name="frog",
    )
    ref_b = predictor.encode_ref(
        Image.new("RGB", (8, 8)),
        box=[2, 2, 7, 7],
        name="leaf",
    )

    out = predictor.predict(Image.new("RGB", (20, 10)), [ref_a, ref_b])

    assert set(out) == {"frog", "leaf"}
    assert len(model.decodes) == 2
    assert out["frog"]["boxes"].shape == (1, 4)
    assert out["frog"]["masks"].shape == (1, 10, 20)
    assert out["frog"]["logits"].shape == (1, 2, 2)
    assert out["frog"]["scores"].tolist() == [float(torch.sigmoid(torch.tensor(2.0)))]


def test_ground_predictor_reranks_reference_candidates_by_similarity():
    model = FakeRefModel()
    predictor = GroundPredictor(
        model,
        device="cpu",
        top_k=1,
        sim_thr=0.5,
    )
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :4] = True
    ref = predictor.encode_ref(Image.new("RGB", (8, 8)), mask=mask, name="frog")

    out = predictor.predict(Image.new("RGB", (8, 8)), [ref])["frog"]

    assert out["masks"].shape == (1, 8, 8)
    assert out["scores"].tolist() == [float(torch.sigmoid(torch.tensor(1.0)))]
    assert out["similarities"][0] > 0.9


def test_ground_predictor_keeps_fixed_image_options():
    predictor = GroundPredictor(FakeGroundModel(), device="cpu")
    assert not hasattr(predictor, "max_masks")

    for kwargs in ({"image_size": 512}, {"max_masks": 1}):
        try:
            GroundPredictor(FakeGroundModel(), device="cpu", **kwargs)
        except TypeError:
            pass
        else:
            raise AssertionError(f"Expected TypeError for {kwargs}")

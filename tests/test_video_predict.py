import numpy as np
import torch
from PIL import Image
from src.predict.video import VideoPredictor
from src.predict.video_ops import session


class FakeVideoModel(torch.nn.Module):
    image_size = 16

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_image(self, image, **kwargs):
        self.calls.append(("forward_image", image.shape, kwargs))
        return {"features": image}

    def init_state(self, **kwargs):
        self.calls.append(("init_state", kwargs))
        return {
            "cached_features": dict(kwargs["cached_features"]),
            "num_frames": kwargs["num_frames"],
            "video_height": kwargs["video_height"],
            "video_width": kwargs["video_width"],
        }

    def add_new_masks(self, state, frame_idx, obj_ids, masks):
        self.calls.append(("add_new_masks", frame_idx, obj_ids, masks.shape))
        state["obj_ids"] = obj_ids
        state["ref_masks"] = masks

    def propagate_in_video_preflight(self, state, run_mem_encoder=True):
        self.calls.append(("preflight", run_mem_encoder))
        state["preflight"] = True

    def propagate_in_video(self, state, **kwargs):
        self.calls.append(("propagate", kwargs, dict(state["cached_features"])))
        low_res = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]])
        video_res = torch.full((1, 1, 6, 8), -1.0)
        video_res[:, :, 2:4, 3:5] = 2.0
        scores = torch.tensor([[2.5]])
        yield kwargs["start_frame_idx"], [7], low_res, video_res, scores


def test_video_predictor_starts_tracker_state():
    model = FakeVideoModel()
    predictor = VideoPredictor(model, {"device": "cpu"})
    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True

    state = predictor.start(Image.new("RGB", (4, 5)), mask, obj_id=7)

    assert state["obj_id"] == 7
    assert state["next_frame"] == 1
    assert state["state"]["num_frames"] == 1
    assert state["state"]["video_height"] == 5
    assert state["state"]["video_width"] == 4
    assert state["state"]["ref_masks"].shape == (1, 4, 4)
    assert ("preflight", True) in model.calls
    assert model.calls[0][2]["need_sam3_out"] is False


def test_video_predictor_propagates_with_tracker_state():
    model = FakeVideoModel()
    predictor = VideoPredictor(model, {"device": "cpu"})
    state = predictor.start(Image.new("RGB", (4, 5)), np.ones((5, 4), dtype=bool))

    out = predictor.predict(Image.new("RGB", (8, 6)), state)

    assert state["next_frame"] == 2
    assert out["frame_idx"] == 1
    assert out["obj_ids"] == [7]
    assert out["masks"].shape == (1, 6, 8)
    assert out["masks"][0, 2:4, 3:5].all()
    assert out["scores"].tolist() == [2.5]
    assert out["logits"].shape == (1, 2, 2)
    assert state["state"]["num_frames"] == 2
    assert 1 in state["state"]["cached_features"]
    assert model.calls[-2][2]["need_sam3_out"] is False


def test_format_output_accepts_bfloat16_tensors():
    low_res = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], dtype=torch.bfloat16)
    video_res = torch.full((1, 1, 6, 8), -1.0, dtype=torch.bfloat16)
    video_res[:, :, 2:4, 3:5] = 2.0
    scores = torch.tensor([[2.5]], dtype=torch.bfloat16)

    out = session.format_output((1, [7], low_res, video_res, scores), 0.0)

    assert out["masks"][0, 2:4, 3:5].all()
    assert out["scores"].dtype == np.float32
    assert out["logits"].dtype == np.float32

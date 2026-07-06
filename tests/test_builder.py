import torch
from src.build import build_image_model
from src.model.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel
from torch import nn


def test_build_image_model_returns_image_model():
    model = build_image_model({"path": None, "device": torch.device("cpu")})

    assert isinstance(model, Sam3ImageModel)
    assert model.training is True


class FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        images,
        need_sam3=True,
        need_interactive=True,
        need_propagation=True,
    ):
        self.calls.append(
            {
                "images": images,
                "need_sam3": need_sam3,
                "need_interactive": need_interactive,
                "need_propagation": need_propagation,
            }
        )
        return {"sam3": {"features": images}}


class FakeVideoVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(
        self,
        images,
        need_sam3=True,
        need_interactive=True,
        need_propagation=True,
    ):
        self.calls.append(
            {
                "images": images,
                "need_sam3": need_sam3,
                "need_interactive": need_interactive,
                "need_propagation": need_propagation,
            }
        )
        return {"propagation": {"features": images}}


class FakeVideoFeat(nn.Module):
    def forward(self, features):
        return {"frame": features}


class FakeVideoMem(nn.Module):
    def forward(self, frame, mask, obj_id=None):
        return {"memory": frame, "mask": mask, "obj_id": obj_id}


class FakeVideoTrack(nn.Module):
    def forward(self, frame, memory, multimask=False):
        return {"track": frame, "memory": memory, "multimask": multimask}


class FakeTrackMgr(nn.Module):
    def forward(self, track, ground, track_ids=None, memory=None, state=None):
        return {
            "managed": track,
            "ground": ground,
            "track_ids": track_ids,
            "memory": memory,
            "state": state,
        }


class FakeGroundImage(nn.Module):
    def forward(self, features):
        return {"image": features}


class FakeGroundPrompt(nn.Module):
    def forward(self, image, **kwargs):
        return {"prompt": kwargs, "image": image}


class FakeGroundDec(nn.Module):
    def forward(self, image, cond, prompt):
        return {"pred_logits": image, "pred_boxes": cond, "pred_masks": prompt}


class FakeCond(nn.Module):
    def __init__(self, value=None):
        super().__init__()
        self.value = {"language_features": "visual"} if value is None else value

    def forward(self):
        return self.value


def test_grounding_model_connects_blocks():
    model = Sam3GroundingModel.__new__(Sam3GroundingModel)
    nn.Module.__init__(model)
    model.vision = FakeVision()
    model.cond = FakeCond()
    model.ground_image = FakeGroundImage()
    model.ground_prompt = FakeGroundPrompt()
    model.ground_dec = FakeGroundDec()

    out = model(
        images="image",
        boxes="boxes",
        box_labels="labels",
    )

    assert model.vision.calls == [
        {
            "images": "image",
            "need_sam3": True,
            "need_interactive": False,
            "need_propagation": False,
        }
    ]
    assert out["pred_logits"] == {"image": {"features": "image"}}
    assert out["pred_boxes"] == {"language_features": "visual"}
    assert out["pred_masks"]["prompt"]["boxes"] == "boxes"
    assert out["pred_masks"]["prompt"]["box_labels"] == "labels"


def test_video_model_connects_blocks():
    model = Sam3VideoModel.__new__(Sam3VideoModel)
    nn.Module.__init__(model)
    model.vision = FakeVideoVision()
    model.video_feat = FakeVideoFeat()
    model.video_mem = FakeVideoMem()
    model.video_track = FakeVideoTrack()
    model.track_mgr = FakeTrackMgr()

    out = model(
        reference_images="ref",
        reference_mask="mask",
        next_images="next",
        obj_id=7,
        multimask=True,
        ground={"pred_logits": "ground"},
    )

    assert model.vision.calls == [
        {
            "images": "ref",
            "need_sam3": False,
            "need_interactive": False,
            "need_propagation": True,
        },
        {
            "images": "next",
            "need_sam3": False,
            "need_interactive": False,
            "need_propagation": True,
        },
    ]
    assert out["track"]["track"] == {"frame": {"features": "next"}}
    assert out["track"]["memory"]["obj_id"] == 7
    assert out["manager"]["ground"] == {"pred_logits": "ground"}


def test_grounding_model_skips_visual_token_when_path_is_none(monkeypatch):
    from src.model import model as model_module

    calls = []

    class Empty(nn.Module):
        pass

    class LoadCond(nn.Module):
        def from_ckpt(self, ckpt):
            self.ckpt = ckpt
            return self

        def forward(self):
            return self.ckpt

    def load_visual(path):
        calls.append(path)
        return {"language_features": torch.ones(1), "language_mask": torch.zeros(1)}

    monkeypatch.setattr(model_module, "VisionCore", Empty)
    monkeypatch.setattr(model_module, "GroundImage", Empty)
    monkeypatch.setattr(model_module, "GroundPrompt", Empty)
    monkeypatch.setattr(model_module, "GroundDec", Empty)
    monkeypatch.setattr(model_module, "VisualCond", LoadCond)
    monkeypatch.setattr(model_module, "load_visual", load_visual)

    model = Sam3GroundingModel()

    assert calls == []
    assert not hasattr(model.cond, "ckpt")


def test_grounding_model_accepts_visual_token_path(monkeypatch):
    from pathlib import Path

    from src.model import model as model_module

    calls = []

    class Empty(nn.Module):
        pass

    class LoadCond(nn.Module):
        def from_ckpt(self, ckpt):
            self.ckpt = ckpt
            return self

    def load_visual(path):
        calls.append(path)
        return {"language_features": torch.ones(1), "language_mask": torch.zeros(1)}

    monkeypatch.setattr(model_module, "VisionCore", Empty)
    monkeypatch.setattr(model_module, "GroundImage", Empty)
    monkeypatch.setattr(model_module, "GroundPrompt", Empty)
    monkeypatch.setattr(model_module, "GroundDec", Empty)
    monkeypatch.setattr(model_module, "VisualCond", LoadCond)
    monkeypatch.setattr(model_module, "load_visual", load_visual)

    Sam3GroundingModel(visual_path=Path("weight/custom_visual.pt"))

    assert calls == [Path("weight/custom_visual.pt")]


def test_image_model_loads_path_with_from_ckpt(monkeypatch):
    from src.model import model as model_module

    calls = []

    class Block(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def from_ckpt(self, ckpt, strict=False):
            calls.append((self.name, ckpt, strict))
            return self

    class FakeCheckpoint:
        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return "ckpt"

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "VisionCore", lambda: Block("vision"))
    monkeypatch.setattr(model_module, "SamImage", lambda: Block("sam_image"))
    monkeypatch.setattr(model_module, "SamPrompt", lambda: Block("sam_prompt"))
    monkeypatch.setattr(model_module, "SamMask", lambda: Block("sam_mask"))

    Sam3ImageModel("model.pt")

    assert calls == [
        ("load", "model.pt"),
        ("vision", "ckpt", False),
        ("sam_image", "ckpt", False),
        ("sam_prompt", "ckpt", False),
        ("sam_mask", "ckpt", False),
    ]


def test_grounding_model_loads_path_with_from_ckpt(monkeypatch):
    from src.model import model as model_module

    calls = []

    class Block(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def from_ckpt(self, ckpt, strict=False):
            calls.append((self.name, ckpt, strict))
            return self

    class Empty(nn.Module):
        pass

    class LoadCond(nn.Module):
        def from_ckpt(self, ckpt):
            return self

    class FakeCheckpoint:
        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return "ckpt"

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "load_visual", lambda path: {})
    monkeypatch.setattr(model_module, "VisualCond", LoadCond)
    monkeypatch.setattr(model_module, "VisionCore", lambda: Block("vision"))
    monkeypatch.setattr(model_module, "GroundImage", Empty)
    monkeypatch.setattr(model_module, "GroundPrompt", lambda: Block("ground_prompt"))
    monkeypatch.setattr(model_module, "GroundDec", lambda: Block("ground_dec"))

    Sam3GroundingModel("model.pt")

    assert calls == [
        ("load", "model.pt"),
        ("vision", "ckpt", False),
        ("ground_prompt", "ckpt", False),
        ("ground_dec", "ckpt", False),
    ]


def test_video_model_loads_path_with_from_ckpt(monkeypatch):
    from src.model import model as model_module

    calls = []

    class Block(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def from_ckpt(self, ckpt, strict=False):
            calls.append((self.name, ckpt, strict))
            return self

    class Empty(nn.Module):
        pass

    class FakeCheckpoint:
        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return "ckpt"

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "VisionCore", lambda: Block("vision"))
    monkeypatch.setattr(model_module, "VideoFeat", Empty)
    monkeypatch.setattr(model_module, "VideoMem", lambda: Block("video_mem"))
    monkeypatch.setattr(model_module, "VideoTrack", lambda: Block("video_track"))
    monkeypatch.setattr(model_module, "TrackMgr", Empty)

    Sam3VideoModel("model.pt")

    assert calls == [
        ("load", "model.pt"),
        ("vision", "ckpt", False),
        ("video_mem", "ckpt", False),
        ("video_track", "ckpt", False),
    ]


def test_build_functions_load_models(monkeypatch):
    import src.build as build_module

    class FakeModel(nn.Module):
        def __init__(self, path=None):
            super().__init__()
            self.path = path

    class FakeGroundingModel(FakeModel):
        def __init__(self, path=None, visual_path=None):
            super().__init__(path)
            self.visual_path = visual_path

    monkeypatch.setattr(build_module, "Sam3ImageModel", FakeModel)
    monkeypatch.setattr(build_module, "Sam3GroundingModel", FakeGroundingModel)
    monkeypatch.setattr(build_module, "Sam3VideoModel", FakeModel)

    image = build_module.build_image_model(
        {"path": "image.pt", "device": torch.device("cpu")}
    )
    grounding = build_module.build_grounding_model(
        {
            "path": "ground.pt",
            "visual_path": "visual.pt",
            "device": torch.device("cpu"),
        }
    )
    video = build_module.build_video_model(
        {"path": "video.pt", "device": torch.device("cpu")}
    )

    assert image.path == "image.pt"
    assert image.training is True
    assert grounding.path == "ground.pt"
    assert grounding.visual_path == "visual.pt"
    assert grounding.training is True
    assert video.path == "video.pt"
    assert video.training is True

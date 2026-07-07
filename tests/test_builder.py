import torch
from src.build import build_image_model
from src.ml.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel
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


class FakeVideo(nn.Module):
    image_size = 16

    def __init__(self):
        super().__init__()
        self.calls = []

    def from_ckpt(self, ckpt, strict=False):
        self.calls.append(("from_ckpt", ckpt, strict))
        return self

    def init_state(self, *args, **kwargs):
        self.calls.append(("init_state", args, kwargs))
        return {"state": "video"}

    def add_new_masks(self, *args, **kwargs):
        self.calls.append(("add_new_masks", args, kwargs))
        return {"masks": "added"}

    def propagate_in_video_preflight(self, *args, **kwargs):
        self.calls.append(("preflight", args, kwargs))
        return {"preflight": True}

    def propagate_in_video(self, *args, **kwargs):
        self.calls.append(("propagate", args, kwargs))
        return iter(("frame_out",))

    def forward_image(self, *args, **kwargs):
        self.calls.append(("forward_image", args, kwargs))
        return {"image": "features"}

    def forward(self, *args, **kwargs):
        self.calls.append(("forward", args, kwargs))
        return {"forward": "out"}


class FakeVideoFeatBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_image(self, *args, **kwargs):
        self.calls.append(("forward_image", args, kwargs))
        return {"sam2_backbone_out": "image_features"}

    def forward(self, features):
        self.calls.append(("forward", features))
        return {"frame": features}


class FakeVideoMemBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.encoder = object()

    def forward(self, frame, mask, obj_id=None):
        self.calls.append((frame, mask, obj_id))
        return {"memory": frame, "mask": mask, "obj_id": obj_id}


class FakeVideoTrackBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def make_tracker(self, video_feat, video_mem):
        self.calls.append(("make_tracker", video_feat, video_mem))
        return FakeVideo()

    def forward(self, frame, memory, multimask=True):
        self.calls.append((frame, memory, multimask))
        masks = torch.zeros(1, 16, 3, 2, 2)
        masks[:, 0, 2] = 2
        iou = torch.tensor([[[0.1, 0.2, 0.9]] + [[0.0, 0.0, 0.0]] * 15])
        scores = torch.ones(1, 16, 1)
        return {
            "propagated_mask_logits": masks,
            "obj_scores": scores,
            "raw": {"iou_pred": iou},
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
    model.runtime = FakeVideo()

    assert model.image_size == 16
    assert model.init_state(device="cpu") == {"state": "video"}
    assert model.add_new_masks("state", masks="mask") == {"masks": "added"}
    assert model.propagate_in_video_preflight("state") == {"preflight": True}
    assert list(model.propagate_in_video("state")) == ["frame_out"]
    assert model.forward_image("image") == {"image": "features"}
    assert model("batch") == {"forward": "out"}
    assert model.runtime.calls == [
        ("init_state", (), {"device": "cpu"}),
        ("add_new_masks", ("state",), {"masks": "mask"}),
        ("preflight", ("state",), {}),
        ("propagate", ("state",), {}),
        ("forward_image", ("image",), {}),
        ("forward", ("batch",), {}),
    ]


def test_video_model_assembles_video_blocks(monkeypatch):
    from src.ml import model as model_module

    monkeypatch.setattr(model_module, "VideoFeat", FakeVideoFeatBlock)
    monkeypatch.setattr(model_module, "VideoMem", FakeVideoMemBlock)
    monkeypatch.setattr(model_module, "VideoTrack", FakeVideoTrackBlock)
    monkeypatch.setattr(
        model_module, "TrackMgr", TrackMgr := type("TrackMgr", (nn.Module,), {})
    )

    model = Sam3VideoModel()

    assert isinstance(model.video_feat, FakeVideoFeatBlock)
    assert isinstance(model.video_mem, FakeVideoMemBlock)
    assert isinstance(model.video_track, FakeVideoTrackBlock)
    assert isinstance(model.track_mgr, TrackMgr)
    assert isinstance(model.runtime, FakeVideo)
    assert model.video_track.calls == [
        ("make_tracker", model.video_feat, model.video_mem)
    ]


def test_grounding_model_skips_visual_token_when_path_is_none(monkeypatch):
    from src.ml import model as model_module

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

    from src.ml import model as model_module

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
    from src.ml import model as model_module

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
    from src.ml import model as model_module

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
    from src.ml import model as model_module

    calls = []

    class FakeRuntime(nn.Module):
        def load_state_dict(self, state, strict=False):
            calls.append(("load_state_dict", state, strict))

    class FakeTrack(FakeVideoTrackBlock):
        def make_tracker(self, video_feat, video_mem):
            calls.append(("make_tracker", video_feat, video_mem))
            return FakeRuntime()

    class FakeCheckpoint:
        def block_state(self, prefix):
            calls.append(("block_state", prefix))
            return {"w": "state"}

        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return cls()

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "VideoFeat", FakeVideoFeatBlock)
    monkeypatch.setattr(model_module, "VideoMem", FakeVideoMemBlock)
    monkeypatch.setattr(model_module, "VideoTrack", FakeTrack)
    monkeypatch.setattr(model_module, "TrackMgr", type("TrackMgr", (nn.Module,), {}))

    Sam3VideoModel("model.pt")

    assert calls[0][0] == "make_tracker"
    assert calls[1:] == [
        ("load", "model.pt"),
        ("block_state", "video"),
        ("load_state_dict", {"w": "state"}, False),
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

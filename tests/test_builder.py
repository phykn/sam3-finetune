import torch
from src.build import build_finetune_loader, build_image_model, build_finetune_model
from src.finetune.adapter import LoraLinear
from src.finetune.model import FinetuneModel
from src.ml.model import Sam3GroundingModel, Sam3ImageModel, Sam3VideoModel
from torch import nn


def test_build_image_model_returns_image_model():
    model = build_image_model({"path": None, "device": torch.device("cpu")})

    assert isinstance(model, Sam3ImageModel)
    assert model.training is True


def test_image_model_decode_accepts_finetune_options():
    class FakeImageMaskDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(
            self,
            image_embed,
            high_res_features,
            prompt,
            image_pe,
            multimask,
            repeat_image,
            mix=None,
        ):
            self.calls.append(
                {
                    "multimask": multimask,
                    "repeat_image": repeat_image,
                    "mix": mix,
                }
            )
            return "decoded"

    model = Sam3ImageModel.__new__(Sam3ImageModel)
    nn.Module.__init__(model)
    model.sam_mask = FakeImageMaskDecoder()

    out = model.decode_masks(
        torch.zeros(1, 256, 2, 2),
        (torch.zeros(1, 32, 8, 8), torch.zeros(1, 64, 4, 4)),
        (torch.zeros(1, 1, 256), torch.zeros(1, 256, 2, 2)),
        torch.zeros(1, 256, 2, 2),
        multimask=False,
        repeat_image=True,
        cond=1,
        prompt_type="point",
    )

    assert out == "decoded"
    assert model.sam_mask.calls == [
        {
            "multimask": False,
            "repeat_image": True,
            "mix": None,
        }
    ]


def test_build_finetune_model_returns_finetune_model(monkeypatch):
    import src.build as build_module

    class FakeImageModel(nn.Module):
        def __init__(self, path=None):
            super().__init__()
            self.path = path
            self.keep = nn.Linear(3, 3)
            self.sam_mask = nn.Module()
            self.sam_mask.mask_decoder = nn.Module()
            self.sam_mask.mask_decoder.transformer = nn.Module()
            self.sam_mask.mask_decoder.transformer.q_proj = nn.Linear(3, 3)

    monkeypatch.setattr(build_module, "Sam3ImageModel", FakeImageModel)

    model = build_finetune_model(
        {
            "path": "image.pt",
            "device": torch.device("cpu"),
            "num_conditions": 3,
            "num_experts": 2,
            "num_classes": 5,
            "lora_rank": 2,
            "feature_rank": 2,
        }
    )

    assert isinstance(model, FinetuneModel)
    assert model.model.path == "image.pt"
    assert model.router.cond.num_embeddings == 3
    assert model.class_head.out_features == 5
    assert isinstance(
        model.model.sam_mask.mask_decoder.transformer.q_proj,
        LoraLinear,
    )


def test_build_finetune_loader_uses_split_and_rank(monkeypatch):
    import src.build as build_module

    calls = []

    def make_loader(
        config,
        num_classes,
        train,
        rank=0,
        world_size=1,
    ):
        calls.append(
            {
                "config": config,
                "num_classes": num_classes,
                "train": train,
                "rank": rank,
                "world_size": world_size,
            }
        )
        return "loader"

    monkeypatch.setattr(build_module, "make_finetune_loader", make_loader)

    loader = build_finetune_loader(
        {
            "paths": ["data/a.json", "data/b.json"],
            "batch_size": 2,
            "conds": [0, 1],
            "labels": [
                {"target": [1, 0], "weight": [1, 1]},
                {"target": [0, 0], "weight": [1, 0]},
            ],
            "num_workers": 0,
        },
        num_classes=2,
        train=False,
        rank=1,
        world_size=2,
    )

    assert loader == "loader"
    assert calls == [
        {
            "config": {
                "paths": ["data/a.json", "data/b.json"],
                "batch_size": 2,
                "conds": [0, 1],
                "labels": [
                    {"target": [1, 0], "weight": [1, 1]},
                    {"target": [0, 0], "weight": [1, 0]},
                ],
                "num_workers": 0,
            },
            "num_classes": 2,
            "train": False,
            "rank": 1,
            "world_size": 2,
        }
    ]


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

    def load_weights(self, ckpt, strict=False):
        self.calls.append(("from_ckpt", ckpt, strict))
        return self

    def init_state(self, *args, **kwargs):
        self.calls.append(("init_state", args, kwargs))
        return {"state": "video"}

    def add_masks(self, *args, **kwargs):
        self.calls.append(("add_masks", args, kwargs))
        return {"masks": "added"}

    def remove_objects(self, *args, **kwargs):
        self.calls.append(("remove_objects", args, kwargs))
        return [1], []

    def propagate_in_video_preflight(self, *args, **kwargs):
        self.calls.append(("preflight", args, kwargs))
        return {"preflight": True}

    def propagate_in_video(self, *args, **kwargs):
        self.calls.append(("propagate", args, kwargs))
        return iter(("frame_out",))

    def forward_image(self, *args, **kwargs):
        self.calls.append(("forward_image", args, kwargs))
        return {"image": "features"}


class FakeVideoFeaturesBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_image(self, *args, **kwargs):
        self.calls.append(("forward_image", args, kwargs))
        return {"sam2_backbone_out": "image_features"}

    def forward(self, features):
        self.calls.append(("forward", features))
        return {"frame": features}


class FakeVideoMemoryBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.encoder = object()

    def forward(self, frame, mask, obj_id=None):
        self.calls.append((frame, mask, obj_id))
        return {"memory": frame, "mask": mask, "obj_id": obj_id}


class FakeVideoTrackingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.transformer = object()
        self.image_pe = object()
        self.mask_decoder = object()
        self.output_valid_embed = torch.nn.Parameter(torch.zeros(1))
        self.output_invalid_embed = torch.nn.Parameter(torch.zeros(1))

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


class FakeGroundingImage(nn.Module):
    def forward(self, features):
        return {
            "image": features,
            "vision_features": torch.zeros(1, 1, 1, 1),
            "vision_mask": None,
            "vision_pos_enc": (),
            "backbone_fpn": (),
            "feat_sizes": (),
        }


class FakeGroundingPromptEncoder(nn.Module):
    def forward(self, image, **kwargs):
        return {
            "features": torch.zeros(1, 1, 1),
            "mask": torch.zeros(1, 1, dtype=torch.bool),
            "prompt": kwargs,
            "image": image,
        }


class FakeGroundingDecoder(nn.Module):
    def forward(self, image, cond, prompt):
        return {
            "pred_logits": {"image": image["image"]},
            "pred_boxes": cond,
            "pred_masks": prompt,
        }


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
    model.ground_image = FakeGroundingImage()
    model.ground_prompt = FakeGroundingPromptEncoder()
    model.ground_dec = FakeGroundingDecoder()

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
    assert model.add_masks("state", masks="mask") == {"masks": "added"}
    assert model.remove_objects("state", [2]) == ([1], [])
    assert model.propagate_in_video_preflight("state") == {"preflight": True}
    assert list(model.propagate_in_video("state")) == ["frame_out"]
    assert model.forward_image("image") == {"image": "features"}
    assert model.runtime.calls == [
        ("init_state", (), {"device": "cpu"}),
        ("add_masks", ("state",), {"masks": "mask"}),
        ("remove_objects", ("state", [2]), {}),
        ("preflight", ("state",), {}),
        ("propagate", ("state",), {}),
        ("forward_image", ("image",), {}),
    ]


def test_video_model_assembles_video_blocks(monkeypatch):
    import src.ml.model.video.model as model_module

    calls = []

    def create_runtime(features, memory, tracking):
        calls.append((features, memory, tracking))
        return FakeVideo()

    monkeypatch.setattr(model_module, "VideoFeatures", FakeVideoFeaturesBlock)
    monkeypatch.setattr(model_module, "VideoMemory", FakeVideoMemoryBlock)
    monkeypatch.setattr(model_module, "VideoTracking", FakeVideoTrackingBlock)
    monkeypatch.setattr(model_module, "create_runtime", create_runtime)
    model = Sam3VideoModel()

    assert isinstance(model.video_feat, FakeVideoFeaturesBlock)
    assert isinstance(model.video_mem, FakeVideoMemoryBlock)
    assert isinstance(model.video_track, FakeVideoTrackingBlock)
    assert isinstance(model.runtime, FakeVideo)
    assert calls == [
        (
            model.video_feat,
            model.video_mem,
            model.video_track,
        )
    ]


def test_grounding_model_skips_visual_token_when_path_is_none(monkeypatch):
    import src.ml.model.grounding as model_module

    calls = []

    class Empty(nn.Module):
        pass

    class LoadCond(nn.Module):
        def load_weights(self, ckpt):
            self.ckpt = ckpt
            return self

        def forward(self):
            return self.ckpt

    def load_visual(path):
        calls.append(path)
        return {"language_features": torch.ones(1), "language_mask": torch.zeros(1)}

    monkeypatch.setattr(model_module, "VisionEncoder", Empty)
    monkeypatch.setattr(model_module, "GroundingImage", Empty)
    monkeypatch.setattr(model_module, "GroundingPromptEncoder", Empty)
    monkeypatch.setattr(model_module, "GroundingDecoder", Empty)
    monkeypatch.setattr(model_module, "VisualTokens", LoadCond)
    monkeypatch.setattr(model_module, "load_visual", load_visual)

    model = Sam3GroundingModel()

    assert calls == []
    assert not hasattr(model.cond, "ckpt")


def test_grounding_model_accepts_visual_token_path(monkeypatch):
    from pathlib import Path

    import src.ml.model.grounding as model_module

    calls = []

    class Empty(nn.Module):
        pass

    class LoadCond(nn.Module):
        def load_weights(self, ckpt):
            self.ckpt = ckpt
            return self

    def load_visual(path):
        calls.append(path)
        return {"language_features": torch.ones(1), "language_mask": torch.zeros(1)}

    monkeypatch.setattr(model_module, "VisionEncoder", Empty)
    monkeypatch.setattr(model_module, "GroundingImage", Empty)
    monkeypatch.setattr(model_module, "GroundingPromptEncoder", Empty)
    monkeypatch.setattr(model_module, "GroundingDecoder", Empty)
    monkeypatch.setattr(model_module, "VisualTokens", LoadCond)
    monkeypatch.setattr(model_module, "load_visual", load_visual)

    Sam3GroundingModel(visual_path=Path("weight/custom_visual.pt"))

    assert calls == [Path("weight/custom_visual.pt")]


def test_image_model_loads_path_with_strict_blocks(monkeypatch):
    import src.ml.model.image as model_module

    calls = []

    class Block(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def load_weights(self, ckpt, strict=False):
            calls.append((self.name, ckpt, strict))
            return self

    class FakeCheckpoint:
        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return "ckpt"

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "VisionEncoder", lambda: Block("vision"))
    monkeypatch.setattr(model_module, "ImageFeatures", lambda: Block("sam_image"))
    monkeypatch.setattr(model_module, "ImagePromptEncoder", lambda: Block("sam_prompt"))
    monkeypatch.setattr(model_module, "ImageMaskDecoder", lambda: Block("sam_mask"))

    Sam3ImageModel("model.pt")

    assert calls == [
        ("load", "model.pt"),
        ("vision", "ckpt", False),
        ("sam_image", "ckpt", False),
        ("sam_prompt", "ckpt", False),
        ("sam_mask", "ckpt", False),
    ]


def test_grounding_model_loads_path_with_strict_blocks(monkeypatch):
    import src.ml.model.grounding as model_module

    calls = []

    class Block(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def load_weights(self, ckpt, strict=False):
            calls.append((self.name, ckpt, strict))
            return self

    class Empty(nn.Module):
        pass

    class LoadCond(nn.Module):
        def load_weights(self, ckpt):
            return self

    class FakeCheckpoint:
        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return "ckpt"

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "load_visual", lambda path: {})
    monkeypatch.setattr(model_module, "VisualTokens", LoadCond)
    monkeypatch.setattr(model_module, "VisionEncoder", lambda: Block("vision"))
    monkeypatch.setattr(model_module, "GroundingImage", Empty)
    monkeypatch.setattr(
        model_module, "GroundingPromptEncoder", lambda: Block("ground_prompt")
    )
    monkeypatch.setattr(model_module, "GroundingDecoder", lambda: Block("ground_dec"))

    Sam3GroundingModel("model.pt")

    assert calls == [
        ("load", "model.pt"),
        ("vision", "ckpt", False),
        ("ground_prompt", "ckpt", False),
        ("ground_dec", "ckpt", False),
    ]


def test_video_model_loads_path_with_strict_block(monkeypatch):
    import src.ml.model.video.model as model_module

    calls = []

    class FakeRuntime(nn.Module):
        def load_state_dict(self, state, strict=False):
            calls.append(("load_state_dict", state, strict))

    def create_runtime(features, memory, tracking):
        calls.append(("create_runtime", features, memory, tracking))
        return FakeRuntime()

    class FakeCheckpoint:
        def load_block(self, name, module):
            calls.append(("load_block", name, module))

        @classmethod
        def load(cls, path):
            calls.append(("load", path))
            return cls()

    monkeypatch.setattr(model_module, "Checkpoint", FakeCheckpoint)
    monkeypatch.setattr(model_module, "VideoFeatures", FakeVideoFeaturesBlock)
    monkeypatch.setattr(model_module, "VideoMemory", FakeVideoMemoryBlock)
    monkeypatch.setattr(model_module, "VideoTracking", FakeVideoTrackingBlock)
    monkeypatch.setattr(model_module, "create_runtime", create_runtime)

    model = Sam3VideoModel("model.pt")

    assert calls[0][0] == "create_runtime"
    assert calls[1:] == [
        ("load", "model.pt"),
        ("load_block", "video", model.runtime),
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

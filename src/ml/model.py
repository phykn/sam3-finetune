import torch
from torch import nn

from ..io.checkpoint import Checkpoint, load_visual
from .blocks.cond import VisualCond
from .blocks.ground_dec import GroundDec
from .blocks.ground_image import GroundImage
from .blocks.ground_prompt import GroundPrompt
from .blocks.sam_image import SamImage
from .blocks.sam_mask import SamMask
from .blocks.sam_prompt import SamPrompt
from .blocks.track_mgr import TrackMgr
from .blocks.video_feat import VideoFeat
from .blocks.video_mem import VideoMem
from .blocks.video_track import VideoTrack
from .blocks.vision import VisionCore
from .structures import NestedTensor


class Sam3ImageModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.vision = VisionCore()
        self.sam_image = SamImage()
        self.sam_prompt = SamPrompt()
        self.sam_mask = SamMask()
        if path is not None:
            self.from_ckpt(Checkpoint.load(path))

    def from_ckpt(self, ckpt, strict=False):
        self.vision.from_ckpt(ckpt, strict=strict)
        self.sam_image.from_ckpt(ckpt, strict=strict)
        self.sam_prompt.from_ckpt(ckpt, strict=strict)
        self.sam_mask.from_ckpt(ckpt, strict=strict)
        return self

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        features = self.vision(
            images,
            need_sam3=False,
            need_interactive=True,
            need_propagation=False,
        )
        if features["interactive"] is None:
            raise RuntimeError("image model expected interactive vision features")

        image_embed, high_res = self.sam_image(features["interactive"])
        return {
            "image_embed": image_embed,
            "high_res_features": list(high_res),
        }

    def encode_prompt(self, points=None, boxes=None, masks=None):
        return self.sam_prompt(points=points, boxes=boxes, masks=masks)

    def decode_masks(
        self,
        image_embed: torch.Tensor,
        high_res_features: tuple[torch.Tensor, ...],
        prompt,
        image_pe: torch.Tensor,
        multimask: bool = True,
        repeat_image: bool = False,
    ):
        return self.sam_mask(
            image_embed,
            high_res_features,
            prompt,
            image_pe,
            multimask,
            repeat_image,
        )

    @property
    def prompt_encoder(self):
        return self.sam_prompt.prompt_encoder

    @property
    def mask_decoder(self):
        return self.sam_mask.mask_decoder


class Sam3GroundingModel(nn.Module):
    def __init__(self, path=None, visual_path=None) -> None:
        super().__init__()
        self.vision = VisionCore()
        self.cond = VisualCond()
        if visual_path is not None:
            self.cond.from_ckpt(load_visual(visual_path))
        self.ground_image = GroundImage()
        self.ground_prompt = GroundPrompt()
        self.ground_dec = GroundDec()
        if path is not None:
            self.from_ckpt(Checkpoint.load(path))

    def from_ckpt(self, ckpt, strict=False):
        self.vision.from_ckpt(ckpt, strict=strict)
        self.ground_prompt.from_ckpt(ckpt, strict=strict)
        self.ground_dec.from_ckpt(ckpt, strict=strict)
        return self

    def encode_image(self, images: torch.Tensor) -> dict[str, object]:
        features = self.vision(
            images,
            need_sam3=True,
            need_interactive=False,
            need_propagation=False,
        )
        if features["sam3"] is None:
            raise RuntimeError("grounding model expected sam3 vision features")
        return self.ground_image(features["sam3"])

    def encode_prompt(
        self,
        image,
        prompt=None,
        boxes=None,
        box_labels=None,
        box_mask=None,
        points=None,
        point_labels=None,
        point_mask=None,
        masks=None,
        mask_labels=None,
        mask_mask=None,
    ):
        return self.ground_prompt(
            image,
            prompt=prompt,
            boxes=boxes,
            box_labels=box_labels,
            box_mask=box_mask,
            points=points,
            point_labels=point_labels,
            point_mask=point_mask,
            masks=masks,
            mask_labels=mask_labels,
            mask_mask=mask_mask,
        )

    def decode(self, image, prompt):
        return self.ground_dec(image, self.cond(), prompt)

    def forward(
        self,
        images: torch.Tensor,
        prompt=None,
        boxes=None,
        box_labels=None,
        box_mask=None,
        points=None,
        point_labels=None,
        point_mask=None,
        masks=None,
        mask_labels=None,
        mask_mask=None,
    ):
        image = self.encode_image(images)
        prompt = self.encode_prompt(
            image,
            prompt=prompt,
            boxes=boxes,
            box_labels=box_labels,
            box_mask=box_mask,
            points=points,
            point_labels=point_labels,
            point_mask=point_mask,
            masks=masks,
            mask_labels=mask_labels,
            mask_mask=mask_mask,
        )
        return self.decode(image, prompt)


class Sam3VideoModel(nn.Module):
    def __init__(self, path=None) -> None:
        super().__init__()
        self.video_feat = VideoFeat()
        self.video_mem = VideoMem()
        self.video_track = VideoTrack()
        self.track_mgr = TrackMgr()
        self.runtime = self.video_track.make_tracker(self.video_feat, self.video_mem)
        if path is not None:
            self.from_ckpt(Checkpoint.load(path))

    def from_ckpt(self, ckpt, strict=False):
        self.runtime.load_state_dict(ckpt.block_state("video"), strict=strict)
        return self

    @property
    def image_size(self):
        return self.runtime.image_size

    def init_state(self, *args, **kwargs):
        return self.runtime.init_state(*args, **kwargs)

    def add_new_masks(self, *args, **kwargs):
        return self.runtime.add_new_masks(*args, **kwargs)

    def propagate_in_video_preflight(self, *args, **kwargs):
        return self.runtime.propagate_in_video_preflight(*args, **kwargs)

    def propagate_in_video(self, *args, **kwargs):
        return self.runtime.propagate_in_video(*args, **kwargs)

    def forward_image(self, image, *args, **kwargs):
        if isinstance(image, torch.Tensor):
            image = NestedTensor(image, None)
        return self.runtime.forward_image(image, *args, **kwargs)

    def forward(
        self,
        *args,
        **kwargs,
    ):
        return self.runtime(*args, **kwargs)

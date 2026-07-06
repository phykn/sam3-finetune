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
        self.vision = VisionCore()
        self.video_feat = VideoFeat()
        self.video_mem = VideoMem()
        self.video_track = VideoTrack()
        self.track_mgr = TrackMgr()
        if path is not None:
            self.from_ckpt(Checkpoint.load(path))

    def from_ckpt(self, ckpt, strict=False):
        self.vision.from_ckpt(ckpt, strict=strict)
        self.video_mem.from_ckpt(ckpt, strict=strict)
        self.video_track.from_ckpt(ckpt, strict=strict)
        return self

    def encode_frame(self, images: torch.Tensor) -> dict[str, object]:
        features = self.vision(
            images,
            need_sam3=False,
            need_interactive=False,
            need_propagation=True,
        )
        if features["propagation"] is None:
            raise RuntimeError("video model expected propagation vision features")
        return self.video_feat(features["propagation"])

    def encode_memory(self, frame, reference_mask, obj_id: int | None = None):
        return self.video_mem(frame, reference_mask, obj_id=obj_id)

    def propagate(self, frame, memory, multimask: bool = False):
        return self.video_track(frame, memory, multimask=multimask)

    def update(self, track, ground, track_ids=None, memory=None, state=None):
        return self.track_mgr(
            track,
            ground,
            track_ids=track_ids,
            memory=memory,
            state=state,
        )

    def forward(
        self,
        reference_images: torch.Tensor,
        reference_mask: torch.Tensor,
        next_images: torch.Tensor,
        obj_id: int | None = None,
        multimask: bool = False,
        ground=None,
        track_ids=None,
        state=None,
    ):
        reference_frame = self.encode_frame(reference_images)
        memory = self.encode_memory(reference_frame, reference_mask, obj_id=obj_id)
        next_frame = self.encode_frame(next_images)
        track = self.propagate(next_frame, memory, multimask=multimask)
        out = {
            "reference_frame": reference_frame,
            "next_frame": next_frame,
            "memory": memory,
            "track": track,
        }
        if ground is not None:
            out["manager"] = self.update(
                track,
                ground,
                track_ids=track_ids,
                memory=memory,
                state=state,
            )
        return out

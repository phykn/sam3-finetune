import gc
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn.attention import sdpa_kernel, SDPBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sam3-main"))
sys.path.insert(0, str(ROOT))

import sam3.model.decoder as upstream_decoder  # noqa: E402
from sam3.model.data_misc import FindStage  # noqa: E402
from sam3.model.geometry_encoders import Prompt as UpPrompt  # noqa: E402
from sam3.model.sam3_multiplex_detector import Sam3MultiplexDetector  # noqa: E402
from sam3.model.vl_combiner import SAM3VLBackboneTri  # noqa: E402
from sam3.model_builder import (  # noqa: E402
    _create_dot_product_scoring,
    _create_geometry_encoder,
    _create_multiplex_tri_backbone,
    _create_sam3_transformer,
    _create_segmentation_head,
)
from src.data import ground as ground_data, image as image_data  # noqa: E402
from src.ml.model import Sam3GroundingModel  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
VISUAL = ROOT / "weight" / "visual_token.pt"
IMAGE = ROOT / "asset" / "frog_tgt.jpg"
POINT = np.array([[455.0, 335.0]], dtype=np.float32)
LABEL = np.array([1], dtype=np.int64)


def main():
    patch_upstream_attention()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(IMAGE).convert("RGB")

    src = run_src(image, device)
    clear()
    upstream = run_upstream(image, device)

    print(f"device: {device}")
    print(f"weight: {WEIGHT}")
    print(f"visual: {VISUAL}")
    compare("pred_logits", src["pred_logits"], upstream["pred_logits"])
    compare("pred_boxes", src["pred_boxes"], upstream["pred_boxes"])
    compare("pred_masks", src["pred_masks"], upstream["pred_masks"])
    print(
        f"mask_xor_pixels: {int(np.logical_xor(src['binary'], upstream['binary']).sum())}"
    )


def run_src(image, device):
    model = Sam3GroundingModel(path=WEIGHT, visual_path=VISUAL).to(device).eval()
    tensor, orig_hw = image_data.make_tensor(image, 1008, device)
    points, point_labels = ground_data.build_points(POINT, LABEL, orig_hw, device)

    with torch.inference_mode(), autocast(device):
        out = model(
            tensor,
            points=points,
            point_labels=point_labels,
        )
    copied = copy_out(out)
    del model
    return copied


def run_upstream(image, device):
    model = build_upstream().to(device).eval()
    tensor, orig_hw = image_data.make_tensor(image, 1008, device)
    points, point_labels = ground_data.build_points(POINT, LABEL, orig_hw, device)
    visual = torch.load(VISUAL, map_location=device, weights_only=True)

    prompt = UpPrompt(
        point_embeddings=points,
        point_labels=point_labels,
    )
    find = FindStage(
        img_ids=torch.tensor([0], dtype=torch.long, device=device),
        text_ids=torch.tensor([0], dtype=torch.long, device=device),
        input_boxes=torch.zeros(0, 4, device=device),
        input_boxes_mask=torch.zeros(1, 0, dtype=torch.bool, device=device),
        input_boxes_label=torch.zeros(0, dtype=torch.long, device=device),
        input_points=torch.zeros(0, 2, device=device),
        input_points_mask=torch.zeros(1, 0, dtype=torch.bool, device=device),
    )

    with torch.inference_mode(), autocast(device):
        backbone_out = model.backbone.forward_image(
            tensor,
            need_sam3_out=True,
            need_interactive_out=False,
            need_propagation_out=False,
        )
        backbone_out["language_features"] = visual["language_features"].to(device)
        backbone_out["language_mask"] = visual["language_mask"].to(device)
        out = model.forward_grounding(
            backbone_out=backbone_out,
            find_input=find,
            find_target=None,
            geometric_prompt=prompt,
        )
    copied = copy_out(out)
    del model
    return copied


def build_upstream():
    backbone = SAM3VLBackboneTri(
        scalp=0,
        visual=_create_multiplex_tri_backbone(use_fa3=False, use_rope_real=False),
        text=None,
    )
    model = Sam3MultiplexDetector(
        num_feature_levels=1,
        backbone=backbone,
        transformer=_create_sam3_transformer(use_fa3=False),
        segmentation_head=_create_segmentation_head(use_fa3=False),
        semantic_segmentation_head=None,
        input_geometry_encoder=_create_geometry_encoder(),
        use_early_fusion=True,
        use_dot_prod_scoring=True,
        dot_prod_scoring=_create_dot_product_scoring(),
        supervise_joint_box_scores=True,
        is_multiplex=True,
    )
    raw = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    state = {
        key.removeprefix("detector."): val
        for key, val in raw.get("model", raw).items()
        if key.startswith("detector.")
        and not key.startswith("detector.backbone.language_backbone.")
    }
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing}, unexpected={unexpected}")
    return model


def copy_out(out):
    logits = out["pred_logits"].detach().cpu().float().numpy().copy()
    boxes = out["pred_boxes"].detach().cpu().float().numpy().copy()
    masks = out["pred_masks"].detach().cpu().float().numpy().copy()
    return {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "pred_masks": masks,
        "binary": masks > 0,
    }


def compare(name, src, upstream):
    diff = np.abs(src - upstream)
    print(f"{name}_max_diff: {float(diff.max()):.8f}")
    print(f"{name}_mean_diff: {float(diff.mean()):.8f}")


def patch_upstream_attention():
    def fallback(_backend):
        return sdpa_kernel(
            [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ]
        )

    upstream_decoder.sdpa_kernel = fallback


def autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

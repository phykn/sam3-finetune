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
from sam3.model_builder import build_sam3_multiplex_video_model  # noqa: E402
from src.data import image as image_data, prompt as prompt_data  # noqa: E402
from src.ml.model import Sam3ImageModel  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
IMAGE = ROOT / "asset" / "heli_1.jpg"
POINT = np.array([[910.0, 345.0]], dtype=np.float32)
LABEL = np.array([1], dtype=np.int32)


def main():
    patch_upstream_attention()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(IMAGE).convert("RGB")

    src = run_src(image, device)
    clear()
    upstream = run_upstream(image, device)

    print(f"device: {device}")
    print(f"weight: {WEIGHT}")
    compare("scores", src["scores"], upstream["scores"])
    compare("masks", src["masks"], upstream["masks"])
    print(
        f"mask_xor_pixels: {int(np.logical_xor(src['binary'], upstream['binary']).sum())}"
    )


def run_src(image, device):
    model = Sam3ImageModel(path=WEIGHT).to(device).eval()
    tensor, orig_hw = image_data.make_tensor(image, 1008, device)
    points = prompt_data.build_points(POINT, LABEL, orig_hw, 1008, device)

    with torch.inference_mode(), autocast(device):
        embed = model.encode_image(tensor)
        encoded = model.encode_prompt(points=points, boxes=None, masks=None)
        masks, scores, *_ = model.decode_masks(
            embed["image_embed"],
            tuple(embed["high_res_features"]),
            encoded,
            model.prompt_encoder.get_dense_pe().to(device),
            True,
            True,
        )
    out = copy_masks(masks, scores)
    del model
    return out


def run_upstream(image, device):
    model = build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        device="cpu",
        use_fa3=False,
        use_rope_real=False,
        strict_state_dict_loading=False,
        compile=False,
    )
    raw = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(video_state(raw), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing}, unexpected={unexpected}")

    model.to(device).eval()
    tensor, orig_hw = image_data.make_tensor(image, 1008, device)
    points = prompt_data.build_points(POINT, LABEL, orig_hw, 1008, device)

    with torch.inference_mode(), autocast(device):
        features = model.forward_image(
            tensor,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=False,
        )["interactive"]
        fpn = features["backbone_fpn"]
        image_embed = fpn[-1].tensors
        image_embed = image_embed + model.interactivity_no_mem_embed.view(
            1, -1, 1, 1
        ).to(image_embed)
        high_res = [fpn[0].tensors, fpn[1].tensors]
        sparse, dense = model.interactive_sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=None,
        )
        masks, scores, *_ = model.interactive_sam_mask_decoder(
            image_embeddings=image_embed,
            image_pe=model.interactive_sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=True,
            repeat_image=True,
            high_res_features=high_res,
        )
    out = copy_masks(masks, scores)
    del model
    return out


def video_state(ckpt):
    raw = ckpt.get("model", ckpt)
    state = {}
    for key, val in raw.items():
        if key.startswith("tracker.model."):
            state[key.removeprefix("tracker.model.")] = val
        elif key.startswith("detector.backbone.vision_backbone."):
            tail = key.removeprefix("detector.backbone.vision_backbone.")
            state["backbone.vision_backbone." + tail] = val
    return state


def copy_masks(masks, scores):
    logits = masks.detach().cpu().float().numpy().copy()
    return {
        "masks": logits,
        "scores": scores.detach().cpu().float().numpy().copy(),
        "binary": logits > 0,
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

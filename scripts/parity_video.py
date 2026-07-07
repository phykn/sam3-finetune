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
from src.data import image as image_data  # noqa: E402
from src.ml.model import Sam3ImageModel, Sam3VideoModel  # noqa: E402
from src.ml.structures import NestedTensor  # noqa: E402
from src.predict.single import SinglePredictor  # noqa: E402
from src.predict.video_ops import session  # noqa: E402

WEIGHT = ROOT / "weight" / "sam3.1_multiplex.pt"
FRAMES = [
    ROOT / "asset" / "heli_1.jpg",
    ROOT / "asset" / "heli_2.jpg",
    ROOT / "asset" / "heli_3.jpg",
]
POINT = np.array([[910.0, 345.0]], dtype=np.float32)
LABEL = np.array([1], dtype=np.int32)


def main():
    patch_upstream_attention()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames = [Image.open(path).convert("RGB") for path in FRAMES]
    mask = make_ref_mask(frames[0], device)

    clear()
    src = run_src(frames, mask, device)

    clear()
    upstream = run_upstream(frames, mask, device)

    print(f"device: {device}")
    print(f"weight: {WEIGHT}")
    for src_out, up_out in zip(src, upstream, strict=True):
        compare(src_out, up_out)


def make_ref_mask(image, device):
    model = Sam3ImageModel(path=WEIGHT)
    predictor = SinglePredictor(model, {"device": device})
    out = predictor.predict(image, point_coords=POINT, point_labels=LABEL)
    index = int(np.argmax(out["scores"]))
    mask = out["masks"][index].copy()
    print(f"reference score: {float(out['scores'][index]):.6f}")
    print(f"reference pixels: {int(mask.sum())}")
    del predictor, model
    return mask


def run_src(frames, mask, device):
    model = Sam3VideoModel(path=WEIGHT)
    return run_video(model, frames, mask, device, nested=True, upstream=False)


def run_upstream(frames, mask, device):
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
    model.to(device)
    return run_video(model, frames, mask, device, nested=False, upstream=True)


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


def run_video(model, frames, mask, device, nested, upstream):
    device = torch.device(device)
    model = model.to(device).eval()
    with torch.inference_mode(), autocast(device):
        state = start(model, frames[0], mask, device, nested, upstream)
        outs = [
            copy_out(predict(model, state, frame, device, nested))
            for frame in frames[1:]
        ]
    del model
    return outs


def start(model, image, mask, device, nested, upstream):
    tensor, features, orig_hw = cache_frame(model, image, device, nested)
    kwargs = {
        "video_height": orig_hw[0],
        "video_width": orig_hw[1],
        "num_frames": 1,
        "cached_features": {0: (tensor, features)},
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
    }
    if not upstream:
        kwargs["device"] = device
    state = model.init_state(**kwargs)
    model.add_new_masks(
        state,
        frame_idx=0,
        obj_ids=[1],
        masks=session.mask_tensor(mask, device),
    )
    model.propagate_in_video_preflight(state, run_mem_encoder=True)
    return {"state": state, "next_frame": 1}


def predict(model, state, image, device, nested):
    frame_idx = state["next_frame"]
    tensor, features, _ = cache_frame(model, image, device, nested)
    tracker_state = state["state"]
    tracker_state["cached_features"][frame_idx] = (tensor, features)
    tracker_state["num_frames"] = max(tracker_state["num_frames"], frame_idx + 1)

    result = None
    for result in model.propagate_in_video(
        tracker_state,
        start_frame_idx=frame_idx,
        max_frame_num_to_track=1,
        reverse=False,
        tqdm_disable=True,
        run_mem_encoder=True,
    ):
        pass

    state["next_frame"] = frame_idx + 1
    if result is None:
        raise RuntimeError("video tracker returned no frame output")
    return session.format_output(result, 0.0)


def cache_frame(model, image, device, nested):
    tensor, orig_hw = image_data.make_tensor(image, model.image_size, device)
    sample = NestedTensor(tensor, None) if nested else tensor
    features = model.forward_image(
        sample,
        need_sam3_out=False,
        need_interactive_out=True,
        need_propagation_out=True,
    )
    return tensor, features, orig_hw


def copy_out(out):
    return {
        "frame_idx": out["frame_idx"],
        "obj_ids": list(out["obj_ids"]),
        "masks": out["masks"].copy(),
        "scores": out["scores"].copy(),
        "logits": out["logits"].copy(),
        "low_res": out["raw"]["low_res_masks"].detach().cpu().float().numpy().copy(),
        "video_res": out["raw"]["video_res_masks"]
        .detach()
        .cpu()
        .float()
        .numpy()
        .copy(),
        "obj_scores": out["raw"]["obj_scores"].detach().cpu().float().numpy().copy(),
    }


def compare(src, upstream):
    print(f"\nframe: {src['frame_idx']}")
    print(f"obj_ids_equal: {src['obj_ids'] == upstream['obj_ids']}")
    print_diff("scores", src["scores"], upstream["scores"])
    print_diff("obj_scores", src["obj_scores"], upstream["obj_scores"])
    print_diff("low_res", src["low_res"], upstream["low_res"])
    print_diff("video_res", src["video_res"], upstream["video_res"])
    xor = np.logical_xor(src["masks"], upstream["masks"]).sum()
    print(f"mask_xor_pixels: {int(xor)}")
    print(f"src_pixels: {int(src['masks'].sum())}")
    print(f"upstream_pixels: {int(upstream['masks'].sum())}")


def print_diff(name, a, b):
    diff = np.abs(a - b)
    print(f"{name}_max_diff: {float(diff.max()):.8f}")
    print(f"{name}_mean_diff: {float(diff.mean()):.8f}")


def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


if __name__ == "__main__":
    main()

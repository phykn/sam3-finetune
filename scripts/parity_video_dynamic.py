import numpy as np
import torch
from PIL import Image
from types import MethodType

import parity_video as base
from src.ml.model import Sam3VideoModel
from src.predict.video_ops import session


def main():
    base.patch_upstream_attention()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = [Image.open(path).convert("RGB") for path in base.FRAMES]
    initial = make_masks(frames[0])
    added = make_added_mask(frames[1])

    base.clear()
    src = Sam3VideoModel(path=base.WEIGHT).to(device).eval()
    src_result = run(src, frames, initial, added, device, source=True)

    base.clear()
    upstream = base.build_sam3_multiplex_video_model(
        checkpoint_path=None,
        load_from_HF=False,
        device="cpu",
        use_fa3=False,
        use_rope_real=False,
        strict_state_dict_loading=False,
        compile=False,
    )
    raw = torch.load(base.WEIGHT, map_location="cpu", weights_only=True)
    missing, unexpected = upstream.load_state_dict(base.video_state(raw), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"missing={missing}, unexpected={unexpected}")
    upstream = upstream.to(device).eval()
    upstream._add_output_per_object = MethodType(
        add_output_per_object,
        upstream,
    )
    upstream_result = run(
        upstream,
        frames,
        initial,
        added,
        device,
        source=False,
    )

    print(f"device: {device}")
    print(f"weight: {base.WEIGHT}")
    compare(src_result, upstream_result)


def make_masks(image):
    width, height = image.size
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, height // 4 : height // 2, width // 4 : width // 2] = 1
    masks[1, height // 2 : 3 * height // 4, width // 2 : 3 * width // 4] = 1
    return masks


def make_added_mask(image):
    width, height = image.size
    mask = np.zeros((1, height, width), dtype=np.float32)
    mask[:, height // 3 : 2 * height // 3, width // 3 : 2 * width // 3] = 1
    return mask


def run(model, frames, initial, added, device, *, source):
    with torch.inference_mode(), base.autocast(device):
        state = start(model, frames[0], initial, device, source)
        frame_one = track(model, state, frames[1], device, source)
        add_result = add(model, state, added, device, source)
        frame_two = track(model, state, frames[2], device, source)
        ids, _ = model.remove_objects(state, [2], strict=True)
        stored = stored_output(state, 2)
    return {
        "frame_one": frame_one,
        "add_ids": list(add_result[1]),
        "add_video": add_result[3].detach().cpu().float().numpy(),
        "frame_two": frame_two,
        "remaining_ids": list(ids),
        "stored_masks": stored["pred_masks"].detach().cpu().float().numpy(),
        "stored_scores": stored["object_score_logits"].detach().cpu().float().numpy(),
    }


def start(model, image, masks, device, source):
    tensor, features, orig_hw = base.cache_frame(model, image, device, source)
    kwargs = {
        "video_height": orig_hw[0],
        "video_width": orig_hw[1],
        "num_frames": 1,
        "cached_features": {0: (tensor, features)},
        "offload_video_to_cpu": False,
        "offload_state_to_cpu": False,
    }
    if source:
        kwargs["device"] = device
    state = model.init_state(**kwargs)
    prompt = session.mask_tensor(masks, device)
    if source:
        model.add_masks(state, 0, [1, 2], prompt)
    else:
        model.add_new_masks(state, 0, [1, 2], prompt)
    model.propagate_in_video_preflight(state, run_mem_encoder=True)
    return state


def track(model, state, image, device, source):
    frame_idx = state["num_frames"]
    tensor, features, _ = base.cache_frame(model, image, device, source)
    state["cached_features"][frame_idx] = (tensor, features)
    state["num_frames"] = frame_idx + 1
    kwargs = {
        "start_frame_idx": frame_idx,
        "max_frame_num_to_track": 1,
        "tqdm_disable": True,
        "run_mem_encoder": True,
    }
    if not source:
        kwargs["reverse"] = False
    result = next(model.propagate_in_video(state, **kwargs))
    return {
        "ids": list(result[1]),
        "low_res": result[2].detach().cpu().float().numpy(),
        "video_res": result[3].detach().cpu().float().numpy(),
        "scores": result[4].detach().cpu().float().numpy(),
    }


def add(model, state, masks, device, source):
    prompt = session.mask_tensor(masks, device)
    if source:
        result = model.add_masks(state, 1, [3], prompt)
    else:
        result = model.add_new_masks(state, 1, [3], prompt)
    model.propagate_in_video_preflight(state, run_mem_encoder=True)
    return result


def stored_output(state, frame_idx):
    output = state["output_dict"]
    return output["cond_frame_outputs"].get(
        frame_idx,
        output["non_cond_frame_outputs"].get(frame_idx),
    )


def add_output_per_object(model, state, frame_idx, current_out, storage_key):
    local_indices = current_out.get("local_obj_id_to_idx", state["obj_id_to_idx"])
    for obj_id, obj_idx in state["obj_id_to_idx"].items():
        local_idx = local_indices.get(obj_id)
        if local_idx is None:
            continue
        item = {
            "pred_masks": current_out["pred_masks"][local_idx : local_idx + 1],
            "object_score_logits": current_out["object_score_logits"][
                local_idx : local_idx + 1
            ],
        }
        if model.use_memory_selection:
            item["iou_score"] = current_out["iou_score"][local_idx : local_idx + 1]
        state["output_dict_per_obj"][obj_idx][storage_key][frame_idx] = item


def compare(src, upstream):
    for key in ("frame_one", "frame_two"):
        print(f"\n{key}")
        print(f"obj_ids_equal: {src[key]['ids'] == upstream[key]['ids']}")
        base.print_diff("low_res", src[key]["low_res"], upstream[key]["low_res"])
        base.print_diff(
            "video_res",
            src[key]["video_res"],
            upstream[key]["video_res"],
        )
        base.print_diff("scores", src[key]["scores"], upstream[key]["scores"])
        xor = np.logical_xor(
            src[key]["video_res"] > 0,
            upstream[key]["video_res"] > 0,
        ).sum()
        print(f"mask_xor_pixels: {int(xor)}")

    print(f"\nadd_ids_equal: {src['add_ids'] == upstream['add_ids']}")
    base.print_diff("add_video", src["add_video"], upstream["add_video"])
    print(
        "remaining_ids_equal: " f"{src['remaining_ids'] == upstream['remaining_ids']}"
    )
    base.print_diff("stored_masks", src["stored_masks"], upstream["stored_masks"])
    base.print_diff("stored_scores", src["stored_scores"], upstream["stored_scores"])


if __name__ == "__main__":
    main()

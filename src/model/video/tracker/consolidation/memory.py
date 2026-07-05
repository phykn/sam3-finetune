import torch
import torch.nn.functional as F


def write_reconstructed_scores(
    model,
    inference_state,
    out,
    batch_size,
    obj_scores,
    iou_scores,
    run_mem_encoder,
):
    if not obj_scores and run_mem_encoder:
        run_mem_encoder = False

    if obj_scores:
        out["object_score_logits"] = torch.cat(obj_scores, dim=0)
    else:
        out["object_score_logits"] = torch.zeros(
            (batch_size, 1),
            dtype=torch.float32,
            device=inference_state["device"],
        )

    if model.use_memory_selection:
        out["iou_score"] = torch.cat(iou_scores, dim=0) if iou_scores else None

    out["obj_ptr"] = None
    return run_mem_encoder


def encode_memory(model, inference_state, frame_idx, batch_size, out):
    device = inference_state["device"]
    high_res_masks = F.interpolate(
        out["pred_masks"].to(device, non_blocking=True),
        size=(model.image_size, model.image_size),
        mode="bilinear",
        align_corners=False,
    )
    high_res_masks = model._apply_non_overlapping_constraints(high_res_masks)
    maskmem_features, maskmem_pos_enc, image_features, image_pos_enc = (
        model._run_memory_encoder(
            inference_state=inference_state,
            frame_idx=frame_idx,
            batch_size=batch_size,
            high_res_masks=high_res_masks,
            object_score_logits=out["object_score_logits"],
            is_mask_from_pts=True,
            conditioning_objects=out["conditioning_objects"],
        )
    )
    out["maskmem_features"] = maskmem_features
    out["maskmem_pos_enc"] = maskmem_pos_enc
    out["image_features"] = image_features
    out["image_pos_enc"] = image_pos_enc

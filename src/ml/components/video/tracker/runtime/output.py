def get_sam_values(model, output):
    values = {
        "low_res_multimasks": output["low_res_multimasks"],
        "high_res_multimasks": output["high_res_multimasks"],
        "ious": output["ious"],
        "low_res_masks": output["low_res_masks"],
        "high_res_masks": output["high_res_masks"],
        "object_score_logits": output["object_score_logits"],
        "obj_ptr": None,
    }
    if model.use_obj_ptrs_in_encoder:
        values["obj_ptr"] = output["obj_ptr"]
    return values


def write_initial_multistep(current_out, values):
    current_out["multistep_pred_masks"] = values["low_res_masks"]
    current_out["multistep_pred_masks_high_res"] = values["high_res_masks"]
    current_out["multistep_pred_multimasks"] = [values["low_res_multimasks"]]
    current_out["multistep_pred_multimasks_high_res"] = [values["high_res_multimasks"]]
    current_out["multistep_pred_ious"] = [values["ious"]]
    current_out["multistep_object_score_logits"] = [values["object_score_logits"]]


def write_final_outputs(model, current_out, values, multiplex_state):
    current_out["pred_masks"] = values["low_res_masks"]
    current_out["pred_masks_high_res"] = values["high_res_masks"]
    if model.use_obj_ptrs_in_encoder:
        current_out["obj_ptr"] = multiplex_state.mux(values["obj_ptr"])
    if model.use_memory_selection:
        current_out["object_score_logits"] = values["object_score_logits"]
        iou_score = current_out["multistep_pred_ious"][-1].max(-1)[0]
        current_out["iou_score"] = iou_score
        current_out["eff_iou_score"] = model.score_memory(
            values["object_score_logits"],
            iou_score,
        )
    current_out["object_score_logits"] = values["object_score_logits"]


def encode_memory(
    model,
    *,
    current_out,
    image,
    features,
    values,
    run_mem_encoder,
    multiplex_state,
):
    if not run_mem_encoder or model.num_maskmem <= 0:
        return
    maskmem_features, maskmem_pos_enc = model._encode_new_memory(
        image=image,
        current_vision_feats=features["propagation_vision_feats"],
        feat_sizes=features["propagation_feat_sizes"],
        pred_masks_high_res=values["high_res_masks"],
        object_score_logits=values["object_score_logits"],
        is_mask_from_pts=False,
        conditioning_objects=current_out["conditioning_objects"],
        multiplex_state=multiplex_state,
    )
    current_out["maskmem_features"] = maskmem_features
    current_out["maskmem_pos_enc"] = maskmem_pos_enc


def save_image_features(model, current_out, features):
    if not model.save_image_features:
        return
    current_out["image_features"] = features["propagation_vision_feats"][-1]
    current_out["image_pos_enc"] = features["propagation_vision_pos_embeds"][-1]

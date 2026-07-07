import torch

from ..prompt.sampling import get_next_point
from ..prompt.utils import concat_points


def _choose_objects_to_correct(
    model,
    *,
    is_init_cond_frame,
    point_inputs,
    gt_masks,
):
    interact_with_all = (
        model.rng.random() < model.prob_correct_all_objects_for_train
    ) or (model.force_correct_all_for_conditional_inputs and is_init_cond_frame)

    if interact_with_all:
        num_objects = gt_masks.shape[0]
    elif model.rand_objects_to_correct_for_train:
        num_objects = model.rng2.integers(
            1,
            int(gt_masks.shape[0] * model.ratio_of_objects_to_correct_for_train) + 1,
        )
    else:
        num_objects = max(
            1,
            int(gt_masks.shape[0] * model.ratio_of_objects_to_correct_for_train),
        )

    objects = model.rng2.choice(
        range(gt_masks.shape[0]),
        size=num_objects,
        replace=False,
    ).tolist()

    if point_inputs is not None:
        point_inputs = {
            "point_coords": point_inputs["point_coords"][objects],
            "point_labels": point_inputs["point_labels"][objects],
        }

    return objects, point_inputs


def _check_point_inputs(point_inputs, objects_to_interact):
    if point_inputs is None:
        return

    assert point_inputs["point_coords"].shape[0] == len(objects_to_interact)
    assert point_inputs["point_labels"].shape[0] == len(objects_to_interact)


def _sample_from_gt(model):
    if model.training and model.prob_to_sample_from_gt_for_train > 0:
        return model.rng.random() < model.prob_to_sample_from_gt_for_train
    return False


def _cast_like(src, dst):
    if torch.is_floating_point(src) and src.dtype != dst.dtype:
        return src.to(dtype=dst.dtype)
    return src


def _merge_selected(dst, src, objects_to_interact):
    dst[objects_to_interact] = _cast_like(src, dst)


def _clone_for_training(model, values):
    if not model.training:
        return values

    cloned = {
        "low_res_masks": values["low_res_masks"].clone(),
        "high_res_masks": values["high_res_masks"].clone(),
        "low_res_multimasks": values["low_res_multimasks"].clone(),
        "high_res_multimasks": values["high_res_multimasks"].clone(),
        "ious": values["ious"].clone(),
        "object_score_logits": values["object_score_logits"].clone(),
        "obj_ptr": None,
    }
    if model.use_obj_ptrs_in_encoder:
        cloned["obj_ptr"] = values["obj_ptr"].clone()

    return cloned


def _record_step_outputs(history, values, point_inputs):
    history["pred_masks"].append(values["low_res_masks"])
    history["pred_high_res_masks"].append(values["high_res_masks"])
    history["pred_multimasks"].append(values["low_res_multimasks"])
    history["pred_high_res_multimasks"].append(values["high_res_multimasks"])
    history["pred_ious"].append(values["ious"])
    history["point_inputs"].append(point_inputs)
    history["object_score_logits"].append(values["object_score_logits"])


def _write_history(current_out, history):
    current_out["multistep_pred_masks"] = torch.cat(history["pred_masks"], dim=1)
    current_out["multistep_pred_masks_high_res"] = torch.cat(
        history["pred_high_res_masks"], dim=1
    )
    current_out["multistep_pred_multimasks"] = history["pred_multimasks"]
    current_out["multistep_pred_multimasks_high_res"] = history[
        "pred_high_res_multimasks"
    ]
    current_out["multistep_pred_ious"] = history["pred_ious"]
    current_out["multistep_point_inputs"] = history["point_inputs"]
    current_out["multistep_object_score_logits"] = history["object_score_logits"]


def _update_conditioning_objects(
    model, current_out, objects_to_interact, multiplex_state
):
    if not model.add_all_frames_to_correct_as_cond:
        return

    if objects_to_interact is None:
        current_out["conditioning_objects"].update(
            multiplex_state.get_all_valid_object_idx()
        )
    else:
        current_out["conditioning_objects"].update(set(objects_to_interact))


def _run_correction_step(
    model,
    *,
    is_init_cond_frame,
    point_inputs,
    mask_inputs,
    gt_masks,
    objects_to_interact,
    interactive_vision_feats,
    interactive_feat_sizes,
    interactive_high_res_features,
    propagation_high_res_features,
    multiplex_state,
    values,
):
    pred_for_new_pt = None
    if not _sample_from_gt(model):
        pred_for_new_pt = values["high_res_masks"] > 0

    new_points, new_labels = get_next_point(
        gt_masks=gt_masks[objects_to_interact],
        pred_masks=(
            pred_for_new_pt[objects_to_interact]
            if pred_for_new_pt is not None
            else None
        ),
        method="uniform" if model.training else model.pt_sampling_for_eval,
    )
    point_inputs = concat_points(point_inputs, new_points, new_labels)
    assert values["low_res_masks"].shape[0] > max(objects_to_interact), (
        f"interacting {objects_to_interact} in {values['low_res_masks'].shape}?"
    )

    if model.iter_use_prev_mask_pred:
        mask_inputs = values["low_res_masks"][objects_to_interact]

    multimask_output = model._use_multimask(is_init_cond_frame, point_inputs)
    pix_feat_with_mem = model._get_interactive_pix_mem(
        interactive_vision_feats, interactive_feat_sizes
    )
    sam_outputs = model._forward_sam_heads(
        backbone_features=pix_feat_with_mem,
        point_inputs=point_inputs,
        mask_inputs=mask_inputs,
        interactive_high_res_features=interactive_high_res_features,
        propagation_high_res_features=propagation_high_res_features,
        multimask_output=multimask_output,
        gt_masks=gt_masks,
        objects_to_interact=objects_to_interact,
        multiplex_state=multiplex_state,
    )

    values = _clone_for_training(model, values)
    merge_sam_outputs(model, values, sam_outputs, objects_to_interact)

    return point_inputs, mask_inputs, values


def merge_sam_outputs(model, values, sam_outputs, objects_to_interact):
    for key in (
        "low_res_masks",
        "high_res_masks",
        "low_res_multimasks",
        "high_res_multimasks",
        "ious",
        "object_score_logits",
    ):
        _merge_selected(values[key], sam_outputs[key], objects_to_interact)

    if model.use_obj_ptrs_in_encoder:
        values["obj_ptr"][objects_to_interact] = sam_outputs["obj_ptr"]


def apply_correction_points(
    model,
    *,
    frame_idx,
    is_init_cond_frame,
    current_out,
    point_inputs,
    mask_inputs,
    gt_masks,
    frames_to_add_correction_pt,
    objects_to_interact,
    interactive_vision_feats,
    interactive_feat_sizes,
    interactive_high_res_features,
    propagation_high_res_features,
    multiplex_state,
    values,
):
    if frame_idx not in frames_to_add_correction_pt:
        return point_inputs, mask_inputs, objects_to_interact, values

    assert gt_masks is not None
    assert interactive_vision_feats is not None
    assert interactive_feat_sizes is not None

    history = {
        "pred_masks": [values["low_res_masks"]],
        "pred_high_res_masks": [values["high_res_masks"]],
        "pred_multimasks": [values["low_res_multimasks"]],
        "pred_high_res_multimasks": [values["high_res_multimasks"]],
        "pred_ious": [values["ious"]],
        "point_inputs": [point_inputs],
        "object_score_logits": [values["object_score_logits"]],
    }

    if model.training:
        assert objects_to_interact is None
        objects_to_interact, point_inputs = _choose_objects_to_correct(
            model,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            gt_masks=gt_masks,
        )
    else:
        assert objects_to_interact is not None

    _check_point_inputs(point_inputs, objects_to_interact)

    for _ in range(model.num_correction_pt_per_frame):
        point_inputs, mask_inputs, values = _run_correction_step(
            model,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            gt_masks=gt_masks,
            objects_to_interact=objects_to_interact,
            interactive_vision_feats=interactive_vision_feats,
            interactive_feat_sizes=interactive_feat_sizes,
            interactive_high_res_features=interactive_high_res_features,
            propagation_high_res_features=propagation_high_res_features,
            multiplex_state=multiplex_state,
            values=values,
        )
        _record_step_outputs(history, values, point_inputs)

    _write_history(current_out, history)
    _update_conditioning_objects(
        model, current_out, objects_to_interact, multiplex_state
    )

    return point_inputs, mask_inputs, objects_to_interact, values

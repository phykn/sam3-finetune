try:
    from typing import NotRequired, Required, TypedDict
except ImportError:
    from typing_extensions import NotRequired, Required, TypedDict

import torch


NO_OBJ_SCORE = -1024.0


class SAMOutput(TypedDict, total=True):
    low_res_multimasks: torch.Tensor
    high_res_multimasks: torch.Tensor
    ious: torch.Tensor
    low_res_masks: torch.Tensor
    high_res_masks: torch.Tensor
    object_score_logits: torch.Tensor
    obj_ptr: NotRequired[torch.Tensor]


class StageOutput(TypedDict, total=False):
    conditioning_objects: Required[set[int]]

    pred_masks: torch.Tensor
    pred_masks_high_res: torch.Tensor
    point_inputs: dict[str, torch.Tensor]
    mask_inputs: torch.Tensor
    object_score_logits: torch.Tensor
    obj_ptr: torch.Tensor
    maskmem_features: torch.Tensor
    maskmem_pos_enc: list[torch.Tensor]
    image_features: torch.Tensor
    image_pos_enc: torch.Tensor

    iou_score: torch.Tensor
    eff_iou_score: torch.Tensor

    multistep_pred_masks: torch.Tensor
    multistep_pred_masks_high_res: torch.Tensor
    multistep_pred_multimasks: list[torch.Tensor]
    multistep_pred_multimasks_high_res: list[torch.Tensor]
    multistep_pred_ious: list[torch.Tensor]
    multistep_point_inputs: list[dict]
    multistep_object_score_logits: list[torch.Tensor]

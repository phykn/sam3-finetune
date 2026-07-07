from .order import (
    apply_object_order,
    get_visible_objects_per_frame,
    handle_empty_start_frame,
    prepare_static_object_order,
    prepare_training_object_order,
    prepare_vos_eval_object_order,
    update_dynamic_prompt_targets,
)
from .sampling import get_next_point, sample_box_points


def prepare_prompt_inputs_meta(self, backbone_out, input, start_frame_idx=0):
    backbone_out["gt_masks_per_frame"] = {
        stage_id: targets.segments.unsqueeze(1)  # [B, 1, H_im, W_im]
        for stage_id, targets in enumerate(input.find_targets)
    }
    num_frames = len(input.find_targets)
    backbone_out["num_frames"] = num_frames

    prompt_config = make_prompt_config(self, num_frames)
    use_pt_input = self.rng.random() < prompt_config["prob_to_use_pt_input"]
    prompt_config = sample_prompt_config(self, prompt_config, use_pt_input)
    backbone_out["use_pt_input"] = use_pt_input

    init_cond_frames = sample_init_cond_frames(
        self,
        start_frame_idx,
        num_frames,
        prompt_config["num_init_cond_frames"],
    )
    backbone_out["init_cond_frames"] = init_cond_frames
    backbone_out["frames_not_in_init_cond"] = [
        t for t in range(start_frame_idx, num_frames) if t not in init_cond_frames
    ]
    backbone_out["frames_to_add_correction_pt"] = sample_correction_frames(
        self,
        use_pt_input,
        init_cond_frames,
        backbone_out["frames_not_in_init_cond"],
        prompt_config["num_frames_to_correct"],
        prompt_config["num_init_cond_frames"],
    )

    return backbone_out


def make_prompt_config(self, num_frames):
    if self.training:
        config = {
            "prob_to_use_pt_input": self.prob_to_use_pt_input_for_train,
            "num_frames_to_correct": self.num_frames_to_correct_for_train,
            "rand_frames_to_correct": self.rand_frames_to_correct_for_train,
            "num_init_cond_frames": self.num_init_cond_frames_for_train,
            "rand_init_cond_frames": self.rand_init_cond_frames_for_train,
        }
    else:
        config = {
            "prob_to_use_pt_input": self.prob_to_use_pt_input_for_eval,
            "num_frames_to_correct": self.num_frames_to_correct_for_eval,
            "rand_frames_to_correct": self.rand_frames_to_correct_for_eval,
            "num_init_cond_frames": self.num_init_cond_frames_for_eval,
            "rand_init_cond_frames": self.rand_init_cond_frames_for_eval,
        }

    if num_frames == 1:
        config["prob_to_use_pt_input"] = 1.0
        config["num_frames_to_correct"] = 1
        config["num_init_cond_frames"] = 1
    assert config["num_init_cond_frames"] >= 1
    return config


def sample_prompt_config(self, config, use_pt_input):
    config = config.copy()
    if config["rand_init_cond_frames"] and config["num_init_cond_frames"] > 1:
        config["num_init_cond_frames"] = self.rng.integers(
            1,
            config["num_init_cond_frames"],
            endpoint=True,
        )
    if (
        use_pt_input
        and config["rand_frames_to_correct"]
        and config["num_frames_to_correct"] > config["num_init_cond_frames"]
    ):
        config["num_frames_to_correct"] = self.rng.integers(
            config["num_init_cond_frames"],
            config["num_frames_to_correct"],
            endpoint=True,
        )
    return config


def sample_init_cond_frames(self, start_frame_idx, num_frames, num_init_cond_frames):
    if num_init_cond_frames == 1:
        return [start_frame_idx]

    return [start_frame_idx] + self.rng.choice(
        range(start_frame_idx + 1, num_frames),
        num_init_cond_frames - 1,
        replace=False,
    ).tolist()


def sample_correction_frames(
    self,
    use_pt_input,
    init_cond_frames,
    frames_not_in_init_cond,
    num_frames_to_correct,
    num_init_cond_frames,
):
    if not use_pt_input:
        return []
    if num_frames_to_correct == num_init_cond_frames:
        return init_cond_frames

    assert num_frames_to_correct > num_init_cond_frames
    extra_num = num_frames_to_correct - num_init_cond_frames
    return (
        init_cond_frames
        + self.rng.choice(frames_not_in_init_cond, extra_num, replace=False).tolist()
    )


def prepare_conditional_frames(self, backbone_out):
    init_cond_frames = backbone_out["init_cond_frames"]
    gt_masks_per_frame = backbone_out["gt_masks_per_frame"]
    use_pt_input = backbone_out["use_pt_input"]

    if self.training:
        prob_to_use_box_input = self.prob_to_use_box_input_for_train
    else:
        prob_to_use_box_input = self.prob_to_use_box_input_for_eval

    backbone_out["mask_inputs_per_frame"] = {}
    backbone_out["point_inputs_per_frame"] = {}
    for t in init_cond_frames:
        if not use_pt_input:
            backbone_out["mask_inputs_per_frame"][t] = gt_masks_per_frame[t]
        else:
            use_box_input = self.rng.random() < prob_to_use_box_input
            if use_box_input:
                points, labels = sample_box_points(
                    gt_masks_per_frame[t],
                )
            else:
                points, labels = get_next_point(
                    gt_masks=gt_masks_per_frame[t],
                    pred_masks=None,
                    method=("uniform" if self.training else self.pt_sampling_for_eval),
                )

            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][t] = point_inputs

    return backbone_out


def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
    backbone_out = self._prepare_prompt_inputs_meta(
        backbone_out, input, start_frame_idx
    )
    backbone_out = self._prepare_conditional_frames(backbone_out)
    return backbone_out


def prepare_dynamic_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
    backbone_out = self._prepare_prompt_inputs_meta(
        backbone_out, input, start_frame_idx=start_frame_idx
    )

    num_frames = backbone_out["num_frames"]
    gt_masks_per_frame = backbone_out["gt_masks_per_frame"]
    visible_objects = get_visible_objects_per_frame(
        self, input, gt_masks_per_frame, num_frames
    )

    init_cond_frames = sorted(backbone_out["init_cond_frames"])
    frames_not_in_init_cond = backbone_out["frames_not_in_init_cond"]
    init_cond_frames = handle_empty_start_frame(
        self,
        visible_objects,
        gt_masks_per_frame,
        init_cond_frames,
        frames_not_in_init_cond,
        start_frame_idx,
        num_frames,
    )
    backbone_out["init_cond_frames"] = init_cond_frames

    order = prepare_dynamic_object_order(
        self,
        visible_objects,
        init_cond_frames,
        frames_not_in_init_cond,
        start_frame_idx,
        num_frames,
    )

    apply_object_order(
        gt_masks_per_frame,
        order["object_appearance_order"],
        order["valid_idx_per_frame"],
        start_frame_idx,
        num_frames,
    )
    update_dynamic_prompt_targets(
        input,
        gt_masks_per_frame,
        order["valid_idx_prior_to_each_transition"],
        order["transition_points"],
    )

    backbone_out["valid_idx_per_frame"] = order["valid_idx_per_frame"]
    backbone_out["new_idx_per_transition"] = order["new_idx_per_transition"]
    backbone_out["valid_objects_prior_to_each_transition"] = order[
        "valid_idx_prior_to_each_transition"
    ]
    backbone_out["transition_points"] = set(order["transition_points"])
    backbone_out["gt_masks_per_frame"] = gt_masks_per_frame
    backbone_out["object_appearance_order"] = order["object_appearance_order"]

    return self._prepare_conditional_frames(backbone_out)


def prepare_dynamic_object_order(
    self,
    visible_objects,
    init_cond_frames,
    frames_not_in_init_cond,
    start_frame_idx,
    num_frames,
):
    if self.training and self.enable_dynamic_training:
        order = prepare_training_object_order(
            self,
            visible_objects,
            init_cond_frames,
            frames_not_in_init_cond,
            start_frame_idx,
            num_frames,
        )
    elif self.is_dynamic_vos_evaluation and not self.training:
        order = prepare_vos_eval_object_order(
            visible_objects,
            init_cond_frames,
            start_frame_idx,
            num_frames,
        )
    else:
        order = prepare_static_object_order(
            visible_objects,
            start_frame_idx,
            num_frames,
        )

    (
        object_appearance_order,
        valid_idx_per_frame,
        valid_idx_prior_to_each_transition,
        new_idx_per_transition,
        transition_points,
    ) = order
    return {
        "object_appearance_order": object_appearance_order,
        "valid_idx_per_frame": valid_idx_per_frame,
        "valid_idx_prior_to_each_transition": valid_idx_prior_to_each_transition,
        "new_idx_per_transition": new_idx_per_transition,
        "transition_points": transition_points,
    }

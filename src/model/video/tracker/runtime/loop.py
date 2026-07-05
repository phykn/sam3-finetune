import torch


def forward_static_tracking(
    self,
    backbone_out,
    input,
    return_dict=False,
    objects_to_interact=None,
):
    has_features, backbone_features = prepare_loop_features(self, backbone_out)
    num_frames = backbone_out["num_frames"]
    init_cond_frames = backbone_out["init_cond_frames"]
    correction_frames = backbone_out["frames_to_add_correction_pt"]
    processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

    output_dict = create_loop_output()
    multiplex_state = self.multiplex_controller.get_state(
        backbone_out["gt_masks_per_frame"][0].shape[0],
        device=backbone_out["gt_masks_per_frame"][0].device,
        dtype=torch.float,
        random=self.training,
    )

    for frame_idx in processing_order:
        img_ids = get_single_image_ids(input, frame_idx)
        image, features = get_frame_features(
            self,
            input,
            backbone_features,
            has_features,
            img_ids,
            need_interactive_out=(
                frame_idx in correction_frames or frame_idx in init_cond_frames
            ),
        )
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=frame_idx in init_cond_frames,
            backbone_features_interactive=features.get("interactive"),
            backbone_features_propagation=features.get("sam2_backbone_out"),
            image=image,
            point_inputs=backbone_out["point_inputs_per_frame"].get(frame_idx),
            mask_inputs=backbone_out["mask_inputs_per_frame"].get(frame_idx),
            gt_masks=backbone_out["gt_masks_per_frame"].get(frame_idx),
            frames_to_add_correction_pt=correction_frames,
            output_dict=output_dict,
            num_frames=num_frames,
            multiplex_state=multiplex_state,
            objects_to_interact=objects_to_interact,
        )

        add_as_cond = frame_idx in init_cond_frames or (
            self.add_all_frames_to_correct_as_cond and frame_idx in correction_frames
        )
        store_frame_output(output_dict, frame_idx, current_out, add_as_cond)

    output_dict["multiplex_state"] = multiplex_state
    if return_dict:
        return output_dict
    return ordered_frame_outputs(output_dict, num_frames)


def forward_dynamic_tracking(
    self,
    backbone_out,
    input,
    return_dict=False,
    objects_to_interact=None,
):
    has_features, backbone_features = prepare_loop_features(self, backbone_out)
    num_frames = backbone_out["num_frames"]
    init_cond_frames = backbone_out["init_cond_frames"]
    correction_frames = backbone_out["frames_to_add_correction_pt"]
    processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

    new_idx_per_transition = backbone_out["new_idx_per_transition"]
    valid_objects_prior = backbone_out["valid_objects_prior_to_each_transition"]
    transition_points = backbone_out["transition_points"]

    output_dict = create_loop_output()
    multiplex_state = create_loop_multiplex_state(
        self, backbone_out, processing_order[0]
    )

    for frame_idx in processing_order:
        current_out = track_dynamic_frame(
            self,
            backbone_out,
            input,
            frame_idx=frame_idx,
            has_features=has_features,
            backbone_features=backbone_features,
            init_cond_frames=init_cond_frames,
            correction_frames=correction_frames,
            transition_points=transition_points,
            new_idx_per_transition=new_idx_per_transition,
            valid_objects_prior=valid_objects_prior,
            multiplex_state=multiplex_state,
            output_dict=output_dict,
            num_frames=num_frames,
            objects_to_interact=objects_to_interact,
        )

        add_as_cond = should_store_dynamic_cond(
            self,
            frame_idx,
            init_cond_frames,
            correction_frames,
            transition_points,
        )
        store_frame_output(output_dict, frame_idx, current_out, add_as_cond)

    output_dict["multiplex_state"] = multiplex_state
    return finalize_dynamic_outputs(
        self,
        output_dict,
        num_frames,
        backbone_out,
        input,
        return_dict,
    )


def track_dynamic_frame(
    self,
    backbone_out,
    input,
    *,
    frame_idx,
    has_features,
    backbone_features,
    init_cond_frames,
    correction_frames,
    transition_points,
    new_idx_per_transition,
    valid_objects_prior,
    multiplex_state,
    output_dict,
    num_frames,
    objects_to_interact,
):
    img_ids = get_single_image_ids(input, frame_idx)
    need_interactive = needs_dynamic_interaction(
        frame_idx,
        correction_frames,
        init_cond_frames,
        transition_points,
    )
    image, features = get_frame_features(
        self,
        input,
        backbone_features,
        has_features,
        img_ids,
        need_interactive_out=need_interactive,
    )
    gt_masks, new_masks, new_idxs = split_transition_masks(
        backbone_out,
        frame_idx,
        new_idx_per_transition,
        valid_objects_prior,
        transition_points,
    )
    return self.track_step(
        frame_idx=frame_idx,
        is_init_cond_frame=frame_idx in init_cond_frames,
        backbone_features_interactive=features.get("interactive"),
        backbone_features_propagation=features.get("sam2_backbone_out"),
        image=image,
        point_inputs=backbone_out["point_inputs_per_frame"].get(frame_idx),
        mask_inputs=backbone_out["mask_inputs_per_frame"].get(frame_idx),
        gt_masks=gt_masks,
        frames_to_add_correction_pt=correction_frames,
        output_dict=output_dict,
        num_frames=num_frames,
        multiplex_state=multiplex_state,
        objects_to_interact=objects_to_interact,
        new_object_masks=new_masks,
        new_object_idxs=new_idxs,
    )


def create_loop_multiplex_state(self, backbone_out, frame_idx):
    gt_masks = backbone_out["gt_masks_per_frame"][frame_idx]
    return self.multiplex_controller.get_state(
        gt_masks.shape[0],
        device=gt_masks.device,
        dtype=torch.float,
        random=self.training,
    )


def needs_dynamic_interaction(
    frame_idx,
    correction_frames,
    init_cond_frames,
    transition_points,
):
    return (
        frame_idx in correction_frames
        or frame_idx in init_cond_frames
        or frame_idx in transition_points
    )


def should_store_dynamic_cond(
    self,
    frame_idx,
    init_cond_frames,
    correction_frames,
    transition_points,
):
    return (
        frame_idx in init_cond_frames
        or (self.add_all_frames_to_correct_as_cond and frame_idx in correction_frames)
        or (self.add_all_transition_frames_as_cond and frame_idx in transition_points)
    )


def finalize_dynamic_outputs(
    self,
    output_dict,
    num_frames,
    backbone_out,
    input,
    return_dict,
):
    if return_dict:
        return output_dict

    outputs = ordered_frame_outputs(
        output_dict,
        num_frames,
        allow_missing=self.is_dynamic_vos_evaluation,
    )
    if self.is_dynamic_vos_evaluation:
        pad_dynamic_vos_outputs(outputs, backbone_out, input)
    return outputs


def prepare_loop_features(self, backbone_out):
    has_features = "interactive" in backbone_out or "sam2_backbone_out" in backbone_out
    if not has_features:
        return False, None
    return True, self._prepare_backbone_features(backbone_out)


def create_loop_output():
    return {
        "cond_frame_outputs": {},
        "non_cond_frame_outputs": {},
    }


def get_single_image_ids(input, frame_idx):
    img_ids = input.find_inputs[frame_idx].img_ids
    assert all(img_id == img_ids[0] for img_id in img_ids)
    return torch.tensor([img_ids[0]], device=img_ids.device, dtype=img_ids.dtype)


def get_frame_features(
    self,
    input,
    backbone_features,
    has_features,
    img_ids,
    *,
    need_interactive_out,
):
    if not has_features:
        return self._prepare_backbone_features_per_frame(
            input.img_batch,
            img_ids,
            need_interactive_out=need_interactive_out,
            need_propagation_out=True,
        )

    image = input.img_batch.tensors[img_ids]
    return image, slice_backbone_features(backbone_features, img_ids)


def slice_backbone_features(backbone_features, img_ids):
    current = {}
    for neck_key, neck_out in backbone_features.items():
        current[neck_key] = {
            "vision_feats": [x[:, img_ids] for x in neck_out["vision_feats"]],
            "vision_masks": [
                x[img_ids] if x is not None else None for x in neck_out["vision_masks"]
            ],
            "vision_pos_embeds": [x[:, img_ids] for x in neck_out["vision_pos_embeds"]],
            "feat_sizes": neck_out["feat_sizes"],
        }
    return current


def split_transition_masks(
    backbone_out,
    frame_idx,
    new_idx_per_transition,
    valid_objects_prior,
    transition_points,
):
    gt_masks = backbone_out["gt_masks_per_frame"].get(frame_idx)
    if frame_idx not in transition_points:
        return gt_masks, None, None

    assert gt_masks is not None
    new_idxs = new_idx_per_transition[frame_idx]
    assert sorted(new_idxs) == new_idxs
    assert new_idxs[0] == len(
        valid_objects_prior[frame_idx]
    ), f"{new_idxs=}; {gt_masks.shape=}; {valid_objects_prior[frame_idx]=}"
    assert new_idxs[-1] == len(gt_masks) - 1, f"{new_idxs=}; {gt_masks.shape=}"
    return gt_masks[: new_idxs[0]], gt_masks[new_idxs], new_idxs


def store_frame_output(output_dict, frame_idx, current_out, add_as_cond):
    if add_as_cond:
        output_dict["cond_frame_outputs"][frame_idx] = current_out
    else:
        output_dict["non_cond_frame_outputs"][frame_idx] = current_out


def ordered_frame_outputs(output_dict, num_frames, allow_missing=False):
    all_outputs = {}
    all_outputs.update(output_dict["cond_frame_outputs"])
    all_outputs.update(output_dict["non_cond_frame_outputs"])

    if allow_missing:
        outputs = [all_outputs.get(frame_idx) for frame_idx in range(num_frames)]
    else:
        outputs = [all_outputs[frame_idx] for frame_idx in range(num_frames)]

    return [
        (
            {key: value for key, value in output.items() if key != "obj_ptr"}
            if output is not None
            else None
        )
        for output in outputs
    ]


def pad_dynamic_vos_outputs(outputs, backbone_out, input):
    object_order = backbone_out["object_appearance_order"]
    num_objects = len(input.find_metadatas[0].coco_image_id)

    inverse_order = [None for _ in object_order]
    for idx, obj_id in enumerate(object_order):
        inverse_order[obj_id] = idx
    assert all(idx is not None for idx in inverse_order)

    if len(inverse_order) < num_objects:
        inverse_order.extend(range(len(inverse_order), num_objects))

    last_mask = outputs[-1]["pred_masks"]
    shape = last_mask.shape[1:]
    dtype = last_mask.dtype
    device = last_mask.device

    for frame_idx, frame_out in enumerate(outputs):
        if frame_out is None:
            outputs[frame_idx] = {
                "pred_masks": torch.zeros(
                    (num_objects, *shape),
                    device=device,
                    dtype=dtype,
                )
            }
            continue

        pred_mask = frame_out["pred_masks"]
        if pred_mask.shape[0] >= num_objects:
            continue

        shape = pred_mask.shape[1:]
        padding = torch.zeros(
            (num_objects - pred_mask.shape[0], *shape),
            device=device,
            dtype=dtype,
        )
        frame_out["pred_masks"] = torch.cat([pred_mask, padding], dim=0)[inverse_order]

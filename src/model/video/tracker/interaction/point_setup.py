import torch

from ..prompt.utils import concat_points


def prepare_point_inputs(
    self,
    inference_state,
    *,
    obj_idx,
    frame_idx,
    points,
    labels,
    clear_old_points,
    rel_coordinates,
):
    point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]

    if points.dim() == 2:
        points = points.unsqueeze(0)
    if labels.dim() == 1:
        labels = labels.unsqueeze(0)

    if rel_coordinates:
        points = points * self.image_size

    points = points.to(inference_state["device"])
    labels = labels.to(inference_state["device"])

    old_point_inputs = None
    if not clear_old_points:
        old_point_inputs = point_inputs_per_frame.get(frame_idx, None)

    point_inputs = concat_points(old_point_inputs, points, labels)
    point_inputs_per_frame[frame_idx] = point_inputs

    return point_inputs


def get_point_frame_context(self, inference_state, frame_idx):
    is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
    reverse = False
    if not is_init_cond_frame:
        reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]

    is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
    storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

    return {
        "is_init_cond_frame": is_init_cond_frame,
        "reverse": reverse,
        "is_cond": is_cond,
        "storage_key": storage_key,
    }


def ensure_point_multiplex_state(self, inference_state, obj_ids):
    multiplex_state = inference_state["multiplex_state"]
    is_new_state = multiplex_state is None

    if is_new_state:
        multiplex_state = self.multiplex_controller.get_state(
            num_valid_entries=1,
            device=inference_state["device"],
            dtype=torch.float32,
            random=False,
            object_ids=obj_ids,
        )
        inference_state["multiplex_state"] = multiplex_state

    return multiplex_state, is_new_state


def get_point_case(*, multiplex_state, is_new_state, obj_id, is_init_cond_frame):
    is_existing_object = (
        not is_new_state
        and multiplex_state is not None
        and obj_id in multiplex_state.object_ids
    )

    if not is_existing_object:
        return "new_object"
    if is_init_cond_frame:
        return "gap_fill"
    return "refine"


def run_new_point_object(
    self,
    inference_state,
    *,
    frame_idx,
    point_inputs,
    obj_idxs,
    obj_ids,
    is_new_state,
):
    current_out, _ = self._run_single_frame_inference(
        inference_state=inference_state,
        output_dict=inference_state["output_dict"],
        frame_idx=frame_idx,
        batch_size=1,
        is_init_cond_frame=True,
        point_inputs=point_inputs,
        mask_inputs=None,
        reverse=False,
        run_mem_encoder=False,
        prev_sam_mask_logits=None,
        add_to_existing_state=not is_new_state,
        new_obj_idxs=obj_idxs,
        new_obj_ids=obj_ids,
        allow_new_buckets=True,
        prefer_new_buckets=True,
        objects_to_interact=None,
    )
    return current_out

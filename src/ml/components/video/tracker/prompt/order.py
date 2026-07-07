import logging
from collections import defaultdict

import torch


def get_visible_objects_per_frame(self, input, gt_masks_per_frame, num_frames):
    if self.training or self.is_dynamic_vos_evaluation:
        return input.visible_objects_per_frame

    return {
        stage_id: set(range(gt_masks_per_frame[stage_id].shape[0]))
        for stage_id in range(num_frames)
    }


def handle_empty_start_frame(
    self,
    visible_objects_per_frame,
    gt_masks_per_frame,
    init_cond_frames,
    frames_not_in_init_cond,
    start_frame_idx,
    num_frames,
):
    if len(visible_objects_per_frame[start_frame_idx]) > 0:
        return init_cond_frames

    if self.training:
        logging.warning("Empty first frame, tracking an empty object")
        visible_objects_per_frame[start_frame_idx] = {0}
        for stage_id in range(num_frames):
            gt_masks_per_frame[stage_id][0] = torch.zeros_like(
                gt_masks_per_frame[stage_id][0]
            )
        return init_cond_frames

    assert self.is_dynamic_vos_evaluation, f"{visible_objects_per_frame=} invalid"
    assert len(init_cond_frames) == 1
    for stage_id in range(start_frame_idx, num_frames):
        if len(visible_objects_per_frame[stage_id]) > 0:
            init_cond_frames = [stage_id]
            break
    for stage_id in range(init_cond_frames[0] + 1):
        if stage_id in frames_not_in_init_cond:
            frames_not_in_init_cond.remove(stage_id)
    return init_cond_frames


def prepare_training_object_order(
    self,
    visible_objects_per_frame,
    init_cond_frames,
    frames_not_in_init_cond,
    start_frame_idx,
    num_frames,
):
    valid_idx_per_frame: dict[int, list[int]] = {}
    valid_idx_prior_to_each_transition: dict[int, list[int]] = {}
    transition_points = choose_transition_points(self, frames_not_in_init_cond)
    transition_points, new_idx_per_transition = filter_new_object_transitions(
        visible_objects_per_frame,
        init_cond_frames,
        transition_points,
        start_frame_idx,
        num_frames,
    )

    init_objects = get_initial_objects(
        self, visible_objects_per_frame, init_cond_frames
    )

    object_appearance_order = init_objects.copy()
    valid_idx_per_frame[start_frame_idx] = list(range(len(init_objects)))
    for stage_id in range(start_frame_idx + 1, num_frames):
        if stage_id in transition_points:
            stage_objects = new_idx_per_transition[stage_id].copy()
            self.rng2.shuffle(stage_objects)
            valid_idx_prior_to_each_transition[stage_id] = list(
                range(len(object_appearance_order))
            )
            new_idx_per_transition[stage_id] = list(
                range(
                    len(object_appearance_order),
                    len(object_appearance_order) + len(stage_objects),
                )
            )
            object_appearance_order.extend(stage_objects)

        if stage_id in init_cond_frames:
            valid_idx_per_frame[stage_id] = valid_idx_per_frame[start_frame_idx].copy()
        elif stage_id in frames_not_in_init_cond:
            valid_idx_per_frame[stage_id] = list(range(len(object_appearance_order)))
        else:
            raise ValueError(
                f"Unexpected {stage_id=}? {init_cond_frames=} {frames_not_in_init_cond=} {transition_points=}"
            )

    return (
        object_appearance_order,
        valid_idx_per_frame,
        valid_idx_prior_to_each_transition,
        new_idx_per_transition,
        transition_points,
    )


def choose_transition_points(self, frames_not_in_init_cond):
    if self.rand_num_transition_points:
        num_transition_points = self.rng.integers(
            1,
            self.max_num_transition_points,
            endpoint=True,
        )
    else:
        num_transition_points = self.max_num_transition_points

    num_transition_points = min(num_transition_points, len(frames_not_in_init_cond))
    transition_points = self.rng2.choice(
        frames_not_in_init_cond,
        num_transition_points,
        replace=False,
    ).tolist()
    return sorted(transition_points)


def filter_new_object_transitions(
    visible_objects_per_frame,
    init_cond_frames,
    transition_points,
    start_frame_idx,
    num_frames,
):
    filtered_transition_points = []
    new_idx_per_transition: dict[int, list[int]] = {}
    objects_seen = set()

    for stage_id in init_cond_frames:
        objects_seen.update(visible_objects_per_frame[stage_id])

    for stage_id in range(start_frame_idx, num_frames):
        if stage_id not in transition_points:
            continue

        new_objects_seen = visible_objects_per_frame[stage_id] - objects_seen
        if len(new_objects_seen) == 0:
            continue

        filtered_transition_points.append(stage_id)
        objects_seen.update(new_objects_seen)
        new_idx_per_transition[stage_id] = list(new_objects_seen)

    return filtered_transition_points, new_idx_per_transition


def get_initial_objects(self, visible_objects_per_frame, init_cond_frames):
    init_objects = set()
    for stage_id in init_cond_frames:
        init_objects.update(visible_objects_per_frame[stage_id])

    init_objects = list(init_objects)
    self.rng2.shuffle(init_objects)
    return init_objects


def prepare_vos_eval_object_order(
    visible_objects_per_frame,
    init_cond_frames,
    start_frame_idx,
    num_frames,
):
    valid_idx_per_frame: dict[int, list[int]] = {}
    valid_idx_prior_to_each_transition: dict[int, list[int]] = {}
    new_idx_per_transition: dict[int, list[int]] = {}

    object_appearance_order: list[int] = []
    object_appear_at_stage: dict[int, int] = {}
    transition_points: list[int] = []
    stage_to_new_objects: dict[int, list[int]] = defaultdict(list)
    for stage_id in range(start_frame_idx, num_frames):
        visible_objects = sorted(list(visible_objects_per_frame[stage_id]))
        for obj_id in visible_objects:
            if obj_id in object_appear_at_stage:
                continue

            object_appear_at_stage[obj_id] = stage_id
            object_appearance_order.append(obj_id)
            stage_to_new_objects[stage_id].append(obj_id)
            if stage_id not in init_cond_frames:
                transition_points.append(stage_id)

    objects_seen_so_far = []
    for stage_id in range(start_frame_idx, num_frames):
        if stage_id in transition_points:
            new_objects = stage_to_new_objects[stage_id]
            num_objects_before = len(objects_seen_so_far)
            valid_idx_prior_to_each_transition[stage_id] = list(
                range(num_objects_before)
            )
            new_idx_per_transition[stage_id] = list(
                range(num_objects_before, num_objects_before + len(new_objects))
            )
            objects_seen_so_far.extend(new_objects)

        if stage_id in init_cond_frames:
            valid_idx_per_frame[stage_id] = list(
                range(len(stage_to_new_objects[stage_id]))
            )
            objects_seen_so_far.extend(stage_to_new_objects[stage_id])
        else:
            valid_idx_per_frame[stage_id] = list(range(len(objects_seen_so_far)))

    return (
        object_appearance_order,
        valid_idx_per_frame,
        valid_idx_prior_to_each_transition,
        new_idx_per_transition,
        transition_points,
    )


def prepare_static_object_order(visible_objects_per_frame, start_frame_idx, num_frames):
    transition_points = []
    visible_objects_on_first_frame = sorted(
        list(visible_objects_per_frame[start_frame_idx])
    )
    object_orderings = list(range(len(visible_objects_on_first_frame)))
    object_appearance_order = visible_objects_on_first_frame.copy()
    valid_idx_per_frame = {
        stage_id: object_orderings.copy()
        for stage_id in range(start_frame_idx, num_frames)
    }

    return object_appearance_order, valid_idx_per_frame, {}, {}, transition_points


def apply_object_order(
    gt_masks_per_frame,
    object_appearance_order,
    valid_idx_per_frame,
    start_frame_idx,
    num_frames,
):
    for stage_id in range(start_frame_idx, num_frames):
        gt_masks_per_frame[stage_id] = gt_masks_per_frame[stage_id][
            object_appearance_order
        ][valid_idx_per_frame[stage_id]]


def update_dynamic_prompt_targets(
    input,
    gt_masks_per_frame,
    valid_idx_prior_to_each_transition,
    transition_points,
):
    for stage_id, targets in enumerate(input.find_targets):
        if stage_id in transition_points:
            prev_objects = valid_idx_prior_to_each_transition[stage_id]
            targets.segments = gt_masks_per_frame[stage_id][prev_objects].squeeze(1)
        else:
            targets.segments = gt_masks_per_frame[stage_id].squeeze(1)
        targets.num_boxes = targets.num_boxes[: targets.segments.shape[0]]

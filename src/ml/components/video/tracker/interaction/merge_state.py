import torch

from ..state import create_output_store


def register_object(inference_state, obj_id):
    obj_idx = len(inference_state["obj_ids"])

    inference_state["obj_ids"].append(obj_id)
    inference_state["obj_id_to_idx"][obj_id] = obj_idx
    inference_state["obj_idx_to_id"][obj_idx] = obj_id
    inference_state["output_dict_per_obj"][obj_idx] = create_output_store()
    inference_state["temp_output_dict_per_obj"][obj_idx] = create_output_store()

    return obj_idx


def _is_empty_multiplex_state(state):
    if state is None:
        return True

    assignments = getattr(state, "assignments", None)
    if assignments is None:
        return True

    return getattr(state, "total_valid_entries", 0) == 0


def sync_multiplex_state(model, inference_state, obj_id, obj_idx):
    state = inference_state.get("multiplex_state")

    if not _is_empty_multiplex_state(state) and getattr(state, "object_ids", None):
        if obj_id in state.object_ids:
            old_idx = state.object_ids.index(obj_id)
            state.remove_objects(object_indices=[old_idx], strict=False)

    if _is_empty_multiplex_state(state):
        inference_state["multiplex_state"] = model.multiplex_controller.get_state(
            num_valid_entries=len(inference_state["obj_ids"]),
            device=inference_state["device"],
            dtype=torch.float32,
            random=False,
            object_ids=list(inference_state["obj_ids"]),
        )
        return inference_state["multiplex_state"]

    state.add_objects(
        object_indices=[obj_idx],
        object_ids=[obj_id],
        allow_new_buckets=True,
    )
    return state


def _copy_singleton_bucket(inference_state, singleton_state, key, obj_idx):
    singleton_obj_idx = 0
    if key not in singleton_state:
        return
    if singleton_obj_idx not in singleton_state[key]:
        return
    if key not in inference_state:
        inference_state[key] = {}

    inference_state[key][obj_idx] = singleton_state[key][singleton_obj_idx].copy()


def copy_singleton_object_state(inference_state, singleton_state, obj_idx):
    _copy_singleton_bucket(
        inference_state,
        singleton_state,
        "point_inputs_per_obj",
        obj_idx,
    )
    _copy_singleton_bucket(
        inference_state,
        singleton_state,
        "mask_inputs_per_obj",
        obj_idx,
    )
    _copy_singleton_bucket(
        inference_state,
        singleton_state,
        "output_dict_per_obj",
        obj_idx,
    )
    _copy_singleton_bucket(
        inference_state,
        singleton_state,
        "temp_output_dict_per_obj",
        obj_idx,
    )

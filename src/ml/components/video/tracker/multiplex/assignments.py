import logging

import torch

PADDING_NUM = -1
_REMOVED_NUM = -1116


logger = logging.getLogger(__name__)


def prepare_object_insert(object_indices, object_ids, state_object_ids):
    object_indices = object_indices.copy()
    assert (object_ids is None) == (
        state_object_ids is None
    ), "object_ids must either be always given or always omitted"

    if object_ids is not None:
        assert len(object_ids) == len(
            object_indices
        ), "object_ids must have the same length as object_indices"
        object_ids = object_ids.copy()

    assert object_indices == sorted(object_indices), "object_indices must be sorted"
    object_indices.reverse()
    if object_ids is not None:
        object_ids.reverse()

    return object_indices, object_ids


def pop_next_object(object_indices, object_ids, state_object_ids):
    idx = object_indices.pop()
    if object_ids is not None and state_object_ids is not None:
        state_object_ids.append(object_ids.pop())
    return idx


def fill_existing_slots(
    assignments,
    object_indices,
    object_ids,
    state_object_ids,
    allowed_bucket_capacity,
):
    slots_filled = 0
    for bucket in assignments:
        for i in range(allowed_bucket_capacity):
            if bucket[i] != PADDING_NUM:
                continue

            bucket[i] = pop_next_object(object_indices, object_ids, state_object_ids)
            slots_filled += 1
            if len(object_indices) == 0:
                return slots_filled

    return slots_filled


def append_new_buckets(
    assignments,
    object_indices,
    object_ids,
    state_object_ids,
    allowed_bucket_capacity,
    multiplex_count,
):
    buckets_created = 0
    while len(object_indices) > 0:
        bucket = [PADDING_NUM] * multiplex_count
        for i in range(allowed_bucket_capacity):
            if len(object_indices) == 0:
                break
            bucket[i] = pop_next_object(object_indices, object_ids, state_object_ids)
        assignments.append(bucket)
        buckets_created += 1

    return buckets_created


def mark_removed_objects(assignments, object_indices):
    missing_indices = object_indices.copy()
    for bucket_idx, bucket in enumerate(assignments):
        for slot_idx, obj_id in enumerate(bucket):
            if obj_id in missing_indices:
                assignments[bucket_idx][slot_idx] = _REMOVED_NUM
                missing_indices.remove(obj_id)
    return missing_indices


def split_removed_buckets(assignments):
    buckets_to_remove = []
    buckets_to_keep = []
    for bucket_idx, bucket in enumerate(assignments):
        all_removed = all(obj_id in [PADDING_NUM, _REMOVED_NUM] for obj_id in bucket)
        if all_removed:
            buckets_to_remove.append(bucket_idx)
            logger.info(
                f"Bucket {bucket_idx} marked for removal - all objects removed/paddings"
            )
        else:
            buckets_to_keep.append(bucket_idx)

    return buckets_to_remove, buckets_to_keep


def drop_buckets(assignments, buckets_to_remove):
    for bucket_idx in reversed(buckets_to_remove):
        del assignments[bucket_idx]


def remap_assignment_ids(assignments):
    all_positive_ids = {
        obj_id for bucket in assignments for obj_id in bucket if obj_id >= 0
    }
    id_mapping = {
        old_id: new_id for new_id, old_id in enumerate(sorted(all_positive_ids))
    }

    for bucket in assignments:
        for index, obj_id in enumerate(bucket):
            if obj_id >= 0:
                bucket[index] = id_mapping[obj_id]

    return id_mapping


def remap_object_ids(object_ids, id_mapping):
    new_object_ids = [None] * len(id_mapping)
    for old_idx, new_idx in id_mapping.items():
        new_object_ids[new_idx] = object_ids[old_idx]

    assert not any(obj_id is None for obj_id in new_object_ids)
    return new_object_ids


def create_shuffled_assignments(
    num_valid_entries,
    num_buckets,
    true_bucket_capacity,
    random,
):
    ids = torch.cat(
        [
            torch.arange(num_valid_entries, dtype=torch.long),
            torch.tensor(
                [PADDING_NUM]
                * (num_buckets * true_bucket_capacity - num_valid_entries),
                dtype=torch.long,
            ),
        ],
        dim=0,
    )
    if random:
        indices = torch.randperm(ids.shape[0], dtype=torch.long)
        ids = ids[indices]

    return [
        ids[i * true_bucket_capacity : (i + 1) * true_bucket_capacity].tolist()
        for i in range(num_buckets)
    ]


def create_capacity_assignments(
    num_valid_entries,
    num_buckets,
    allowed_bucket_capacity,
    true_bucket_capacity,
    random,
):
    if random:
        ids = torch.randperm(num_valid_entries, dtype=torch.int64)
    else:
        ids = torch.arange(num_valid_entries)

    total_elements = num_buckets * allowed_bucket_capacity
    if ids.shape[0] < total_elements:
        ids = torch.cat(
            [
                ids,
                torch.tensor([PADDING_NUM] * (total_elements - ids.shape[0])),
            ]
        )

    return [
        ids[i * allowed_bucket_capacity : (i + 1) * allowed_bucket_capacity].tolist()
        + [PADDING_NUM] * (true_bucket_capacity - allowed_bucket_capacity)
        for i in range(num_buckets)
    ]

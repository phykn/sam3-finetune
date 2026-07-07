import logging
import math

import torch
from torch import nn

from .assignments import (
    append_new_buckets,
    create_capacity_assignments,
    create_shuffled_assignments,
    drop_buckets,
    fill_existing_slots,
    mark_removed_objects,
    PADDING_NUM,
    prepare_object_insert,
    remap_assignment_ids,
    remap_object_ids,
    split_removed_buckets,
)

logger = logging.getLogger(__name__)


class MultiplexState:
    """
    Tracks object-slot assignments between data space and multiplex bucket space.
    """

    def __init__(
        self,
        assignments: list[list[int]],
        device: torch.device,
        dtype: torch.dtype,
        allowed_bucket_capacity: int,
        *,
        object_ids: list[int] | None = None,
    ):
        self.device = device
        self.dtype = dtype

        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(
        self, assignments: list[list[int]], *, object_ids: list[int] | None = None
    ):
        self.assignments = assignments
        self.num_buckets = len(self.assignments)
        if self.num_buckets == 0:
            logger.error("No buckets found in the state")
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(self.assignments[0])
        assert all(
            len(self.assignments[i]) == self.multiplex_count
            for i in range(self.num_buckets)
        )

        self.total_valid_entries = sum(
            sum(1 for x in bucket if x >= 0) for bucket in self.assignments
        )
        self.total_non_padding_entries = sum(
            sum(1 for x in bucket if x != PADDING_NUM) for bucket in self.assignments
        )

        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries, (
                "object_ids should map 1:1 to the valid entries"
            )

        all_object_idxs = set()
        for bucket in self.assignments:
            valid_entries_in_bucket = sum(1 for x in bucket if x != PADDING_NUM)
            assert valid_entries_in_bucket <= self.allowed_bucket_capacity, (
                f"{valid_entries_in_bucket=} > {self.allowed_bucket_capacity=}"
            )
            for obj_idx in bucket:
                if obj_idx >= 0:
                    assert obj_idx < self.total_non_padding_entries, (
                        f"object ID {obj_idx} >= {self.total_non_padding_entries}"
                    )
                    assert obj_idx not in all_object_idxs, "object IDs must be unique"
                    all_object_idxs.add(obj_idx)

        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        return (
            self.num_buckets * self.allowed_bucket_capacity
            - self.total_non_padding_entries
        )

    def find_next_batch_of_available_indices(
        self,
        num_objects: int,
        *,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ) -> list[int]:
        assert num_objects > 0, f"{num_objects=} must be positive"
        if not allow_new_buckets:
            assert self.available_slots >= num_objects, (
                f"not enough available slots {self.available_slots} < {num_objects}"
            )

        return list(
            range(
                self.total_valid_entries,
                self.total_valid_entries + num_objects,
            )
        )

    def add_objects(
        self,
        object_indices: list[int],
        *,
        object_ids: list[int] | None = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ):
        if len(object_indices) == 0:
            return

        object_indices, object_ids = prepare_object_insert(
            object_indices,
            object_ids,
            self.object_ids,
        )
        num_new_objects = len(object_indices)

        if prefer_new_buckets:
            assert allow_new_buckets, "prefer_new_buckets requires allow_new_buckets"

        slots_filled = 0
        if not prefer_new_buckets:
            slots_filled = fill_existing_slots(
                self.assignments,
                object_indices,
                object_ids,
                self.object_ids,
                self.allowed_bucket_capacity,
            )

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(
                f"Cannot place objects {list(reversed(object_indices))} without creating new buckets"
            )

        buckets_created = append_new_buckets(
            self.assignments,
            object_indices,
            object_ids,
            self.object_ids,
            self.allowed_bucket_capacity,
            self.multiplex_count,
        )

        original_num_entries = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original_num_entries + num_new_objects, (
            f"{self.total_valid_entries=} != {original_num_entries=} + {num_new_objects=}"
        )

        logger.info(
            f"Filled {slots_filled} slots and created {buckets_created} new buckets"
        )
        logger.info(
            f"{self.num_buckets=}, {self.total_valid_entries=}, {self.total_non_padding_entries=}"
        )

    def remove_objects(self, object_indices: list[int], strict: bool = True):
        missing_indices = mark_removed_objects(self.assignments, object_indices)

        if strict:
            assert len(missing_indices) == 0, (
                f"Failed to remove objects: {missing_indices}"
            )

        buckets_to_remove, buckets_to_keep = split_removed_buckets(self.assignments)
        drop_buckets(self.assignments, buckets_to_remove)

        if len(buckets_to_keep) == 0:
            logger.info(f"Removing all buckets: {buckets_to_remove}; state invalidated")
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        id_mapping = remap_assignment_ids(self.assignments)
        if self.object_ids is not None:
            self.object_ids = remap_object_ids(self.object_ids, id_mapping)

        self._initialize_assignments(self.assignments, object_ids=self.object_ids)

        logger.info(f"Removed these buckets: {buckets_to_remove}")
        logger.info(f"Kept these buckets: {buckets_to_keep}")
        logger.info(
            f"Remaining buckets: {self.num_buckets}, total valid entries: {self.total_valid_entries}"
        )

        return buckets_to_keep

    def _precompute_transition_matrices(self, device: torch.device, dtype: torch.dtype):
        self.mux_matrix = torch.zeros(
            self.num_buckets * self.multiplex_count,
            self.total_valid_entries,
            device=device,
            dtype=dtype,
        )

        self.demux_matrix = torch.zeros(
            self.total_valid_entries,
            self.num_buckets * self.multiplex_count,
            device=device,
            dtype=dtype,
        )

        for i in range(self.num_buckets):
            for j in range(self.multiplex_count):
                bucket_idx = i * self.multiplex_count + j
                object_idx = self.assignments[i][j]
                if object_idx >= 0:
                    self.mux_matrix[bucket_idx, object_idx] = 1.0
                    self.demux_matrix[object_idx, bucket_idx] = 1.0

    def mux(self, x: torch.Tensor) -> torch.Tensor:
        num_valid_entries = x.shape[0]
        assert num_valid_entries == self.total_valid_entries, (
            f"{num_valid_entries=} != {self.total_valid_entries=}"
        )
        output_shape = (
            self.num_buckets,
            self.multiplex_count,
        ) + x.shape[1:]

        x_flat = x.reshape(num_valid_entries, -1)

        result_flat = self.mux_matrix @ x_flat

        result = result_flat.view(output_shape)
        return result

    def demux(self, x: torch.Tensor) -> torch.Tensor:
        num_buckets, multiplex_count = x.shape[:2]
        assert num_buckets == self.num_buckets, f"{num_buckets=} != {self.num_buckets=}"
        assert multiplex_count == self.multiplex_count, (
            f"{multiplex_count=} != {self.multiplex_count=}"
        )
        output_shape = (self.total_valid_entries,) + x.shape[2:]

        x_flat = x.reshape(num_buckets * multiplex_count, -1)

        result_flat = self.demux_matrix @ x_flat

        result = result_flat.view(output_shape)
        return result

    def get_valid_object_mask(self) -> torch.Tensor:
        valid_mask = self.mux_matrix.sum(dim=1) > 0
        valid_mask = valid_mask.reshape(self.num_buckets, self.multiplex_count)

        return valid_mask

    def get_all_valid_object_idx(self) -> set[int]:
        all_valid_objects = {
            obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0
        }
        return all_valid_objects


class MultiplexController(nn.Module):
    def __init__(
        self,
        multiplex_count: int,
        full_shuffle: bool = False,
        eval_multiplex_count: int = -1,
    ):
        super().__init__()

        self.multiplex_count = multiplex_count
        self.full_shuffle = full_shuffle
        if eval_multiplex_count < 0:
            self.eval_multiplex_count = multiplex_count
        else:
            self.eval_multiplex_count = eval_multiplex_count
        assert self.multiplex_count >= 1

    @property
    def allowed_bucket_capacity(self) -> int:
        return self.multiplex_count if self.training else self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        device: torch.device,
        dtype: torch.dtype,
        random: bool = True,
        *,
        object_ids: list[int] | None = None,
    ) -> MultiplexState:
        allowed_bucket_capacity = self.allowed_bucket_capacity
        true_bucket_capacity = self.multiplex_count

        num_buckets = math.ceil(num_valid_entries / allowed_bucket_capacity)

        if self.full_shuffle:
            assignments = create_shuffled_assignments(
                num_valid_entries,
                num_buckets,
                true_bucket_capacity,
                random,
            )
        else:
            assignments = create_capacity_assignments(
                num_valid_entries,
                num_buckets,
                allowed_bucket_capacity,
                true_bucket_capacity,
                random,
            )

        return MultiplexState(
            assignments,
            device,
            dtype,
            allowed_bucket_capacity=allowed_bucket_capacity,
            object_ids=object_ids,
        )

import torch


def _is_muxed(tensor, state):
    if tensor is None or state is None:
        return False
    return tensor.shape[:2] == (state.num_buckets, state.multiplex_count)


def demux_if_needed(tensor, state):
    if _is_muxed(tensor, state):
        return state.demux(tensor)
    return tensor


def filled_object_tensor(num_objs, sample, fill_value=0):
    shape = (num_objs, *sample.shape[1:])
    if fill_value == 0:
        return torch.zeros(shape, dtype=sample.dtype, device=sample.device)

    return torch.full(
        shape,
        fill_value,
        dtype=sample.dtype,
        device=sample.device,
    )


def tensor_with_object_row(sample, num_objs, obj_idx, fill_value=0):
    tensor = filled_object_tensor(num_objs, sample, fill_value)
    tensor[obj_idx : obj_idx + 1] = sample
    return tensor


def pad_first_dim(tensor, size, fill_value=0):
    if tensor.shape[0] >= size:
        return tensor

    pad_shape = (size - tensor.shape[0], *tensor.shape[1:])
    if fill_value == 0:
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    else:
        pad = torch.full(
            pad_shape,
            fill_value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    return torch.cat([tensor, pad], dim=0)

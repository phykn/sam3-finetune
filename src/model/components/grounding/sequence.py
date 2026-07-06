import torch


def is_right_padded(mask):
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(left, left_mask, right, right_mask):
    left_len, batch_size, hidden_size = left.shape
    right_len, right_batch_size, right_hidden_size = right.shape

    assert batch_size == right_batch_size == left_mask.size(0) == right_mask.size(0)
    assert hidden_size == right_hidden_size
    assert left_len == left_mask.size(1)
    assert right_len == right_mask.size(1)

    torch._assert_async(is_right_padded(left_mask))
    torch._assert_async(is_right_padded(right_mask))

    left_actual_len = (~left_mask).sum(dim=-1)
    right_actual_len = (~right_mask).sum(dim=-1)
    final_len = left_actual_len + right_actual_len
    max_len = left_len + right_len

    concat_mask = torch.arange(max_len, device=right.device)[None] >= final_len[:, None]
    concat = torch.zeros(
        (max_len, batch_size, hidden_size),
        device=right.device,
        dtype=right.dtype,
    )
    concat[:left_len, :, :] = left

    index = torch.arange(right_len, device=right.device)[:, None]
    index = index + left_actual_len[None]

    concat = concat.scatter(
        0,
        index[:, :, None].expand(-1, -1, hidden_size),
        right,
    )

    return concat, concat_mask

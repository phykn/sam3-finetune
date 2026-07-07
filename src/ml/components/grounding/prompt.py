import torch

from .sequence import concat_padded_sequences


def append_padded_tokens(
    current_tokens, current_labels, current_mask, tokens, labels, mask
):
    assert list(tokens.shape[:2]) == list(labels.shape[:2])

    batch_size = tokens.shape[1]
    if mask is None:
        mask = torch.zeros(
            batch_size,
            tokens.shape[0],
            dtype=torch.bool,
            device=tokens.device,
        )

    if current_tokens is None:
        return tokens, labels, mask

    assert tokens.shape[1] == labels.shape[1] == batch_size
    assert current_tokens.shape[1] == batch_size

    labels, _ = concat_padded_sequences(
        current_labels.unsqueeze(-1),
        current_mask,
        labels.unsqueeze(-1),
        mask,
    )
    labels = labels.squeeze(-1)
    tokens, mask = concat_padded_sequences(current_tokens, current_mask, tokens, mask)

    return tokens, labels, mask


class Prompt:
    """
    box_embeddings: [N_boxes, B, C_box]
    box_mask: [B, N_boxes]
    point_embeddings: [N_points, B, C_point]
    point_mask: [B, N_points]
    mask_embeddings: [N_masks, B, 1, H_mask, W_mask]
    mask_mask: [B, N_masks]
    box_labels: [N_boxes, B]
    point_labels: [N_points, B]
    mask_labels: [N_masks, B]
    """

    def __init__(
        self,
        box_embeddings=None,
        box_mask=None,
        point_embeddings=None,
        point_mask=None,
        box_labels=None,
        point_labels=None,
        mask_embeddings=None,
        mask_mask=None,
        mask_labels=None,
    ):
        if (
            box_embeddings is None
            and point_embeddings is None
            and mask_embeddings is None
        ):
            self._clear()
            return

        box_seq_len, point_seq_len, mask_seq_len, batch_size, device = (
            self._resolve_shape(box_embeddings, point_embeddings, mask_embeddings)
        )

        box_embeddings, box_labels, box_mask = self._init_box(
            box_embeddings,
            box_labels,
            box_mask,
            box_seq_len,
            batch_size,
            device,
        )
        point_embeddings, point_labels, point_mask = self._init_point(
            point_embeddings,
            point_labels,
            point_mask,
            point_seq_len,
            batch_size,
            device,
        )
        mask_embeddings, mask_labels, mask_mask = self._init_mask(
            mask_embeddings,
            mask_labels,
            mask_mask,
            mask_seq_len,
            batch_size,
            device,
        )

        self.box_embeddings = box_embeddings
        self.point_embeddings = point_embeddings
        self.mask_embeddings = mask_embeddings
        self.box_mask = box_mask
        self.point_mask = point_mask
        self.mask_mask = mask_mask
        self.box_labels = box_labels
        self.point_labels = point_labels
        self.mask_labels = mask_labels

        self._check_shapes(box_seq_len, point_seq_len, mask_seq_len, batch_size)
        self._check_devices(device)

    def _clear(self):
        self.box_embeddings = None
        self.box_labels = None
        self.box_mask = None
        self.point_embeddings = None
        self.point_labels = None
        self.point_mask = None
        self.mask_embeddings = None
        self.mask_mask = None
        self.mask_labels = None

    def _check_shapes(self, box_seq_len, point_seq_len, mask_seq_len, batch_size):
        specs = (
            (
                self.box_embeddings,
                [box_seq_len, batch_size],
                "box embeddings",
                f"[{box_seq_len}, {batch_size}, *]",
                True,
            ),
            (
                self.box_mask,
                [batch_size, box_seq_len],
                "box mask",
                f"[{batch_size}, {box_seq_len}]",
                True,
            ),
            (
                self.box_labels,
                [box_seq_len, batch_size],
                "box labels",
                f"[{box_seq_len}, {batch_size}]",
                True,
            ),
            (
                self.point_embeddings,
                [point_seq_len, batch_size],
                "point embeddings",
                f"[{point_seq_len}, {batch_size}, *]",
                True,
            ),
            (
                self.point_mask,
                [batch_size, point_seq_len],
                "point mask",
                f"[{batch_size}, {point_seq_len}]",
                True,
            ),
            (
                self.point_labels,
                [point_seq_len, batch_size],
                "point labels",
                f"[{point_seq_len}, {batch_size}]",
                True,
            ),
            (
                self.mask_embeddings,
                [mask_seq_len, batch_size],
                "mask embeddings",
                f"[{mask_seq_len}, {batch_size}, *]",
                False,
            ),
            (
                self.mask_mask,
                [batch_size, mask_seq_len],
                "mask attn. mask",
                f"[{batch_size}, {mask_seq_len}]",
                False,
            ),
            (
                self.mask_labels,
                [mask_seq_len, batch_size],
                "mask labels",
                f"[{mask_seq_len}, {batch_size}]",
                False,
            ),
        )

        for tensor, expected, name, expected_text, required in specs:
            if tensor is None and not required:
                continue

            actual = None if tensor is None else tensor.shape
            assert (
                tensor is not None and list(tensor.shape[: len(expected)]) == expected
            ), f"Wrong dimension for {name}. Expected {expected_text} got {actual}"

    def _check_devices(self, device):
        tensors = (
            ("box embeddings", self.box_embeddings, True),
            ("box mask", self.box_mask, True),
            ("box labels", self.box_labels, True),
            ("point embeddings", self.point_embeddings, True),
            ("point mask", self.point_mask, True),
            ("point labels", self.point_labels, True),
            ("mask embeddings", self.mask_embeddings, False),
            ("mask attn. mask", self.mask_mask, False),
            ("mask labels", self.mask_labels, False),
        )

        for name, tensor, required in tensors:
            if tensor is None and not required:
                continue
            assert tensor is not None and tensor.device == device, (
                f"Expected {name} to be on device {device}, got "
                f"{None if tensor is None else tensor.device}"
            )

    def _resolve_shape(self, box_embeddings, point_embeddings, mask_embeddings):
        lengths = {
            "box": 0,
            "point": 0,
            "mask": 0,
        }
        batch_size = None
        device = None

        for name, tensor in (
            ("box", box_embeddings),
            ("point", point_embeddings),
            ("mask", mask_embeddings),
        ):
            if tensor is None:
                continue

            lengths[name] = tensor.shape[0]
            if batch_size is None:
                batch_size = tensor.shape[1]
            else:
                assert batch_size == tensor.shape[1], (
                    f"Batch size mismatch for {name} embeddings. Got "
                    f"{batch_size} and {tensor.shape[1]}."
                )

            if device is None:
                device = tensor.device
            else:
                assert (
                    device == tensor.device
                ), f"Device mismatch for {name} embeddings."

        return lengths["box"], lengths["point"], lengths["mask"], batch_size, device

    def _init_box(
        self,
        box_embeddings,
        box_labels,
        box_mask,
        box_seq_len,
        batch_size,
        device,
    ):
        if box_embeddings is None:
            box_embeddings = torch.zeros(box_seq_len, batch_size, 4, device=device)
        if box_labels is None:
            box_labels = torch.ones(
                box_seq_len,
                batch_size,
                device=device,
                dtype=torch.long,
            )
        if box_mask is None:
            box_mask = torch.zeros(
                batch_size,
                box_seq_len,
                device=device,
                dtype=torch.bool,
            )
        return box_embeddings, box_labels, box_mask

    def _init_point(
        self,
        point_embeddings,
        point_labels,
        point_mask,
        point_seq_len,
        batch_size,
        device,
    ):
        if point_embeddings is None:
            point_embeddings = torch.zeros(point_seq_len, batch_size, 2, device=device)
        if point_labels is None:
            point_labels = torch.ones(
                point_seq_len,
                batch_size,
                device=device,
                dtype=torch.long,
            )
        if point_mask is None:
            point_mask = torch.zeros(
                batch_size,
                point_seq_len,
                device=device,
                dtype=torch.bool,
            )
        return point_embeddings, point_labels, point_mask

    def _init_mask(
        self,
        mask_embeddings,
        mask_labels,
        mask_mask,
        mask_seq_len,
        batch_size,
        device,
    ):
        # Mask embeddings keep their own resolution until the mask encoder runs.
        if mask_labels is None:
            mask_labels = torch.ones(
                mask_seq_len,
                batch_size,
                device=device,
                dtype=torch.long,
            )
        if mask_mask is None:
            mask_mask = torch.zeros(
                batch_size,
                mask_seq_len,
                device=device,
                dtype=torch.bool,
            )
        return mask_embeddings, mask_labels, mask_mask

    def append_boxes(self, boxes, labels, mask=None):
        self.box_embeddings, self.box_labels, self.box_mask = append_padded_tokens(
            current_tokens=self.box_embeddings,
            current_labels=self.box_labels,
            current_mask=self.box_mask,
            tokens=boxes,
            labels=labels,
            mask=mask,
        )

    def append_points(self, points, labels, mask=None):
        self.point_embeddings, self.point_labels, self.point_mask = (
            append_padded_tokens(
                current_tokens=self.point_embeddings,
                current_labels=self.point_labels,
                current_mask=self.point_mask,
                tokens=points,
                labels=labels,
                mask=mask,
            )
        )

    def append_masks(self, masks, labels=None, attn_mask=None):
        if labels is not None:
            assert list(masks.shape[:2]) == list(labels.shape[:2])
        if self.mask_embeddings is None:
            self.mask_embeddings = masks
            mask_seq_len, batch_size = masks.shape[:2]
            if labels is None:
                self.mask_labels = torch.ones(
                    mask_seq_len,
                    batch_size,
                    device=masks.device,
                    dtype=torch.long,
                )
            else:
                self.mask_labels = labels
            if attn_mask is None:
                self.mask_mask = torch.zeros(
                    batch_size,
                    mask_seq_len,
                    device=masks.device,
                    dtype=torch.bool,
                )
            else:
                self.mask_mask = attn_mask
        else:
            raise NotImplementedError("Only one mask per prompt is supported.")

    def clone(self):
        return Prompt(
            box_embeddings=(
                None if self.box_embeddings is None else self.box_embeddings.clone()
            ),
            box_mask=None if self.box_mask is None else self.box_mask.clone(),
            point_embeddings=(
                None if self.point_embeddings is None else self.point_embeddings.clone()
            ),
            point_mask=None if self.point_mask is None else self.point_mask.clone(),
            box_labels=None if self.box_labels is None else self.box_labels.clone(),
            point_labels=(
                None if self.point_labels is None else self.point_labels.clone()
            ),
            mask_embeddings=(
                None if self.mask_embeddings is None else self.mask_embeddings.clone()
            ),
            mask_mask=None if self.mask_mask is None else self.mask_mask.clone(),
            mask_labels=None if self.mask_labels is None else self.mask_labels.clone(),
        )

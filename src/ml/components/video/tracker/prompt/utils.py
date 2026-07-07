import torch

from ..outputs import SAMOutput, StageOutput


def concat_points(old_point_inputs, new_points, new_labels):
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


def append_stage_output(
    target: StageOutput,
    source: SAMOutput,
    target_key: str,
    source_key: str,
    dim: int = 0,
    strict: bool = True,
):
    if strict:
        assert target_key in target, f"{target_key} not found"
    elif target_key not in target:
        return

    target[target_key] = torch.cat([target[target_key], source[source_key]], dim=dim)


def merge_stage_output(
    target: StageOutput,
    source: SAMOutput,
    target_key: str,
    source_key: str,
    source_indices: list[int],
    strict: bool = True,
):
    if strict:
        assert target_key in target, f"{target_key} not found"
    elif target_key not in target:
        return
    target[target_key][source_indices] = source[source_key].to(
        dtype=target[target_key].dtype
    )

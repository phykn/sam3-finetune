from collections.abc import Iterator
from contextlib import contextmanager
from enum import auto, Enum


def write_output(out, key, value, auxiliary=True, update_aux=True):
    out[key] = value[-1] if auxiliary else value
    if auxiliary and update_aux:
        if "aux_outputs" not in out:
            out["aux_outputs"] = [{} for _ in range(len(value) - 1)]
        assert len(out["aux_outputs"]) == len(value) - 1
        for aux_output, aux_value in zip(out["aux_outputs"], value[:-1]):
            aux_output[key] = aux_value


def write_box_outputs(
    out,
    scores,
    boxes,
    boxes_xyxy,
    num_o2o,
    num_o2m,
    training,
):
    write_output(
        out,
        "pred_logits",
        scores[:, :, :num_o2o],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes",
        boxes[:, :, :num_o2o],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes_xyxy",
        boxes_xyxy[:, :, :num_o2o],
        update_aux=training,
    )
    if num_o2m <= 0 or not training:
        return

    write_output(
        out,
        "pred_logits_o2m",
        scores[:, :, num_o2o:],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes_o2m",
        boxes[:, :, num_o2o:],
        update_aux=training,
    )
    write_output(
        out,
        "pred_boxes_xyxy_o2m",
        boxes_xyxy[:, :, num_o2o:],
        update_aux=training,
    )


class SAM3Output(list):
    class IterMode(Enum):
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()

    def __init__(
        self,
        output: list[list[dict]] | None = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: list[int] | None = None,
    ):
        if output is not None:
            assert (
                isinstance(output, list)
                and len(output) > 0
                and isinstance(output[0], list)
            ), "Expected output to be a list of lists"
            self.output = output
        else:
            self.output = []
        assert isinstance(
            iter_mode, SAM3Output.IterMode
        ), f"iter_mode should be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"

        self.iter_mode = iter_mode
        self.loss_stages = loss_stages

    def __iter__(self) -> Iterator:
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return iter(self.output)
        if self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return (stage[-1] for stage in self.output)
        if self.iter_mode == SAM3Output.IterMode.FLATTENED:
            return (item for stage in self.output for item in stage)
        raise ValueError(f"unknown iter_mode: {self.iter_mode}")

    def __getitem__(self, index):
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        if self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        if self.iter_mode == SAM3Output.IterMode.FLATTENED:
            if index == -1:
                return self.output[-1][-1]
            flattened_output = sum(self.output, [])
            return flattened_output[index]
        raise ValueError(f"unknown iter_mode: {self.iter_mode}")

    @staticmethod
    @contextmanager
    def iteration_mode(
        model_output: "SAM3Output",
        iter_mode: IterMode,
    ):
        original_iter_mode = model_output.iter_mode
        model_output.iter_mode = iter_mode
        try:
            yield model_output
        finally:
            model_output.iter_mode = original_iter_mode

    def append(self, item: list):
        assert isinstance(
            item, list
        ), f"Only list items are supported. Got {type(item)}"
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in [
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ]:
            return len(self.output)
        if self.iter_mode == SAM3Output.IterMode.FLATTENED:
            flattened_output = sum(self.output, [])
            return len(flattened_output)
        raise ValueError(f"unknown iter_mode: {self.iter_mode}")

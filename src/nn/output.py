import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Dict, List, Optional

from typing_extensions import override


class SAM3Output(list):
    class IterMode(Enum):
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = auto()

    def __init__(
        self,
        output: List[List[Dict]] = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: Optional[List[int]] = None,
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
        assert isinstance(iter_mode, SAM3Output.IterMode), (
            f"iter_mode shoulf be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"
        )

        self.iter_mode = iter_mode
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (
                inner_list[-1] for inner_list in self_ref().output
            ),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }
        self.loss_stages = loss_stages

    @override
    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

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

    class _IterationMode(AbstractContextManager):
        def __init__(
            self,
            model_output: "SAM3Output",
            iter_mode: "SAM3Output.IterMode",
        ):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        @override
        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        @override
        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(
        model_output: "SAM3Output",
        iter_mode: IterMode,
    ) -> _IterationMode:
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(item, list), (
            f"Only list items are supported. Got {type(item)}"
        )
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

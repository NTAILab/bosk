from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from ..data import Data
from ..stages import Stage
from ..slot import BlockInputSlot, BlockOutputSlot, InputSlotMeta, OutputSlotMeta
from ..pipeline import BasePipeline


class BaseExecutor(ABC):
    """Base pipeline executor.
    """

    def __init__(self, pipeline: BasePipeline, *,
                 stage: None | Stage = None,
                 inputs: None | Mapping[str, InputSlotMeta | Sequence[InputSlotMeta]] = None,
                 outputs: None | Mapping[str, OutputSlotMeta] = None):
        assert stage is not None, "Stage must be specified"
        assert inputs is not None, "Inputs must be specified"
        assert outputs is not None, "Outputs must be specified"
        self.pipeline = pipeline
        self.stage = stage
        self.inputs = inputs
        self.outputs = outputs

    @abstractmethod
    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        ...

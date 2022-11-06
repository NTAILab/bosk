from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from ..data import Data
from ..stages import Stage
from ..block.base import BaseBlock, BlockOutputData
from ..slot import BlockInputSlot, BlockOutputSlot
from ..pipeline import BasePipeline

InputSlotToDataMapping = Mapping[BlockInputSlot, Data]
"""Block input slot data mapping.

It is indexed by input slots.
"""

class BaseExecutor(ABC):
    """Base pipeline executor.

    Attributes:
        pipeline: The pipeline (computational graph). Contains blocks as nodes
            and connections between output and input blocks' slots as edges.
        stage: The computational mode, which will be performed by the executor.
        inputs: The dictionary, containing input names as keys and
            block input slots as values. Computational graph's start points.
        outputs: The dictionary, containing output names as keys and
            block output slots as values. Computational graph's end points.
        
    Args:
        pipeline: Sets :attr:`pipeline`.
        stage: Sets :attr:`stage`.
        inputs: Sets :attr:`inputs`.
        outputs: Sets :attr:`outputs`.
    """

    pipeline: BasePipeline
    stage: Stage
    inputs: None | Mapping[str, BlockInputSlot | Sequence[BlockInputSlot]]
    outputs: None | Mapping[str, BlockOutputSlot]

    def __init__(self, pipeline: BasePipeline, *,
                 stage: None | Stage = None,
                 inputs: None | Mapping[str, BlockInputSlot | Sequence[BlockInputSlot]] = None,
                 outputs: None | Mapping[str, BlockOutputSlot] = None):
        assert stage is not None, "Stage must be specified"
        assert inputs is not None, "Inputs must be specified"
        assert outputs is not None, "Outputs must be specified"
        self.pipeline = pipeline
        self.stage = stage
        self.inputs = inputs
        self.outputs = outputs

    def _is_input_slot_required(self, input_slot: BlockInputSlot) -> bool:
        if self.stage == Stage.FIT:
            return input_slot.meta.stages.fit \
                or input_slot.meta.stages.transform \
                or input_slot.meta.stages.transform_on_fit
        elif self.stage == Stage.TRANSFORM:
            return input_slot.meta.stages.transform
        else:
            raise NotImplementedError()

    def _execute_block(self, node: BaseBlock, node_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if self.stage == Stage.FIT:
            node.fit({
                slot.meta.name: values
                for slot, values in node_input_mapping.items()
                if slot.meta.stages.fit
            })
        filtered_node_input_mapping = {
            slot.meta.name: values
            for slot, values in node_input_mapping.items()
            if slot.meta.stages.transform or (self.stage == Stage.FIT and slot.meta.stages.transform_on_fit)
        }
        return node.wrap(node.transform(filtered_node_input_mapping))

    @abstractmethod
    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        ...

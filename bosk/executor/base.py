from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from ..data import Data
from ..stages import Stage
from ..block.base import BaseBlock, BlockOutputData
from ..slot import BlockInputSlot, BlockOutputSlot, BaseSlot
from ..pipeline import BasePipeline

InputSlotToDataMapping = Mapping[BlockInputSlot, Data]
"""Block input slot data mapping.

It is indexed by input slots.
"""

class BaseSlotStrategy(ABC):
    """The interface for classes, parametrizing executor's behaviour
    during the ececution process. Determines slots' handling politics.
    """
    
    @abstractmethod
    def is_slot_required(self, slot: BaseSlot) -> bool:
        """Method that determines if the slot is required during
        the computational graph execution.
        """

class BaseExecutionStrategy(ABC):
    """The interface for classes, parametrizing executor's behaviour
    during the ececution process. Determines blocks' handling politics.
    """
    
    @abstractmethod
    def execute_block(self, block: BaseBlock, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Method that executes the block.
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

    def __init__(self, pipeline: BasePipeline, slots_handler: BaseSlotStrategy,
                 blocks_handler: BaseExecutionStrategy, *,
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
        self.slots_handler = slots_handler
        self.blocks_handler = blocks_handler

    @abstractmethod
    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        ...

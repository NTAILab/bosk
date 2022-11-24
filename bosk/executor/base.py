from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping, Sequence, Set

from ..data import Data
from ..stages import Stage
from ..block.base import BaseBlock, BlockOutputData
from ..slot import BlockInputSlot, BaseSlot
from ..pipeline import BasePipeline
import warnings

InputSlotToDataMapping = Mapping[BlockInputSlot, Data]
"""Block input slot data mapping.

It is indexed by input slots.
"""

class BaseSlotHandler(ABC):
    """The interface for classes, parametrizing executor's behaviour
    during the execution process. Determines slots' handling policy.
    """
    
    @abstractmethod
    def is_slot_required(self, slot: BaseSlot) -> bool:
        """Method that determines if the slot is required during
        the computational graph execution.
        """

class BaseBlockHandler(ABC):
    """The interface for classes, parametrizing executor's behaviour
    during the execution process. Determines blocks' handling policy.
    """
    
    @abstractmethod
    def execute_block(self, block: BaseBlock, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Method that executes the block.
        """

@dataclass(frozen=True)
class BaseExecutor(ABC):
    """Base pipeline executor.

    Attributes:
        pipeline: The pipeline (computational graph). Contains blocks as nodes
            and connections between output and input blocks' slots as edges.
        stage: The computational mode, which will be performed by the executor.
        slots_handler: Object defining the executor's behaviour during slots processing.
        blocks_handler: Object defining the executor's behaviour during blocks processing.
        inputs: Set of the inputs to process. Passing it, you set up the hard requirement 
            for the input values to execute the computational graph. Keep it `None` to
            use any of the pipeline's inputs during the execution process.
        outputs: Set of the outputs to process. Keep it `None` to handle all of the
            pipeline's outputs.
        
    Args:
        pipeline: Sets :attr:`pipeline`.
        stage: Sets :attr:`stage`.
        slots_handler: Sets :attr:`slots_handler`.
        blocks_handler: Sets :attr:`blocks_handler`.
        inputs: Sets :attr:`inputs`.
        outputs: Sets :attr:`outputs`.
    
    Raises:
        AssertionError: If it was unable to find some input in the pipeline.
        AssertionError: If it was unable to find some output in the pipeline.
    """

    pipeline: BasePipeline
    slot_handler: BaseSlotHandler
    block_handler: BaseBlockHandler
    stage: Stage
    inputs: None | Set[str] = None
    outputs: None | Set[str] = None

    def __post_init__(self) -> None:
        if self.inputs is not None:
            for inp in self.inputs:
                assert inp in self.pipeline.inputs, f'Unable to find input "{inp}" in the pipeline'
            self.inputs = set(self.inputs)
        if self.outputs is not None:
            for out in self.outputs:
                assert out in self.pipeline.outputs, f'Unable to find output "{out}" in the pipeline'
            self.outputs = set(self.outputs)

    def _map_input_names_to_slots(self, input_values: Mapping[str, Data]) -> Mapping[BlockInputSlot, Data]:
        """Method to translate dictionary, passed in :meth:`__call__`, to dictionary that is useful for evaluation.
        Args:
            input_values: Input data, passed to the :meth:`__call__` method.
        Returns:
            Remapped input data.
        """
        inp_slots_to_data_map = dict()
        for name, data in input_values.items():
            input_slot = self.pipeline.inputs.get(name, None)
            if input_slot is None:
                continue
            inp_slots_to_data_map[input_slot] = data
        return inp_slots_to_data_map

    def _check_input_values(self, input_values: Mapping[str, Data]) -> None:
        """Method to check up the input values, passed in :meth:`__call__`.
        Raises:
            AssertionError: If the :attr:`inputs` are specified and the :arg:`input_values`
                do not correspond to them.
        """
        for name in input_values:
            if self.inputs is not None:
                assert name in self.inputs, f'Input "{name}" is not in the executor\'s inputs set'
            input_slot = self.pipeline.inputs.get(name, None)
            if input_slot is None:
                warnings.warn(f'Unable to find input "{name}" in the pipeline')

    @abstractmethod
    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        ...

from abc import ABC, abstractmethod
from typing import Dict, Mapping, FrozenSet, Optional, Sequence, Union
import warnings

import numpy as np

from ..data import BaseData, CPUData, Data
from ..stages import Stage
from ..block.base import BaseBlock, BlockOutputData, BlockInputSlot, BaseSlot
from ..pipeline import BasePipeline
from .block import InputSlotToDataMapping, BaseBlockExecutor, DefaultBlockExecutor


class BaseSlotHandler(ABC):
    """Determines slots' handling policy: checks whether a slot is required for the stage.
    """

    @abstractmethod
    def is_slot_required(self, stage: Stage, slot: BaseSlot) -> bool:
        """Method that determines if the slot is required during
        the computational graph execution.

        Args:
            stage: The execution stage.
            slot: The computational block's slot to check.
        """


class DefaultSlotHandler(BaseSlotHandler):
    def is_slot_required(self, stage: Stage, slot: BaseSlot) -> bool:
        assert isinstance(
            slot, BlockInputSlot), "InputSlotStrategy proceeds only input slots"
        if stage == Stage.FIT:
            return slot.meta.stages.fit \
                or slot.meta.stages.transform \
                or slot.meta.stages.transform_on_fit
        return slot.meta.stages.transform


class ExecutionResult(Dict[str, BaseData]):
    """Pipeline execution result. Basically behaves as a dictionary of `BaseData` objects.

    Wraps a dictionary of `BaseData` objects and provides method to obtain dictionary of NumPy arrays.
    Rationale: frequently expected result of the pipeline execution is a dictionary of NumPy arrays.
    """
    def __init__(self, result: Dict[str, BaseData]):
        """Initialize execution result with dictionary of results.

        Args:
            result: Dictionary of result `BaseData` objects.
        """
        super().__init__(result)

    def numpy(self) -> Dict[str, np.ndarray]:
        """Convert execution result to dictionary of NumPy arrays.

        Returns:
            Dictionary of NumPy arrays.
        """
        return {k: v.to_cpu().data for k, v in self.items()}


class BaseExecutor(ABC):
    """Base pipeline executor.

    Attributes:
        __pipeline: The pipeline (computational graph). Contains blocks as nodes
            and connections between output and input blocks' slots as edges.
        __stage: The computational mode, which will be performed by the executor.
        __slots_handler: Object defining the executor's behaviour during slots processing.
        __blocks_handler: Object defining the executor's behaviour during blocks processing.
        __inputs: Set of the inputs to process. Passing it, you set up the hard requirement
            for the input values to execute the computational graph. Keep it `None` to
            use any of the pipeline's inputs during the execution process.
        __outputs: Set of the outputs to process. Keep it `None` to handle all of the
            pipeline's outputs.

    Args:
        pipeline: Sets :attr:`__pipeline`.
        stage: Sets :attr:`__stage`.
        inputs: Sets :attr:`__inputs`.
        outputs: Sets :attr:`__outputs`.
        slot_handler: Sets :attr:`__slot_handler` with `_prepare_slot_handler` method.
        block_executor: Sets :attr:`__block_executor` with `_prepare_block_executor` method.

    Raises:
        AssertionError: If it was unable to find some input in the pipeline.
        AssertionError: If it was unable to find some output in the pipeline.
    """

    __pipeline: BasePipeline
    __slot_handler: BaseSlotHandler
    __block_executor: BaseBlockExecutor
    __stage: Stage
    __inputs: None | FrozenSet[str]
    __outputs: None | FrozenSet[str]

    def __init__(self, pipeline: BasePipeline,
                 stage: Stage,
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None,
                 slot_handler: Optional[BaseSlotHandler] = None,
                 block_executor: Optional[BaseBlockExecutor] = None) -> None:
        self.__pipeline = pipeline
        self.__slot_handler = self._prepare_slot_handler(slot_handler)
        self.__block_executor = self._prepare_block_executor(block_executor)
        self.__stage = stage
        self.__process_inputs_outputs(inputs, outputs)

    def _prepare_slot_handler(self, slot_handler: Optional[BaseSlotHandler]):
        """The default slot handler can be changed in a child class by overriding this method.

        Args:
            slot_handler: Slot handler passed by user or None.

        Returns:
            Slot handler.

        """
        if slot_handler is None:
            return DefaultSlotHandler()
        return slot_handler

    def _prepare_block_executor(self, block_executor: Optional[BaseBlockExecutor]):
        """The default block executor can be changed in a child class by overriding this method.

        Args:
            block_executor: Slot handler passed by user or None.

        Returns:
            Block executor.

        """
        if block_executor is None:
            return DefaultBlockExecutor()
        return block_executor

    def __process_inputs_outputs(self, inputs: Optional[Sequence[str]], outputs: Optional[Sequence[str]]) -> None:
        if inputs is not None:
            for inp in inputs:
                assert inp in self.__pipeline.inputs, f'Unable to find input "{inp}" in the pipeline'
            self.__inputs = frozenset(inputs)
        else:
            self.__inputs = None
        if outputs is not None:
            for out in outputs:
                assert out in self.__pipeline.outputs, f'Unable to find output "{out}" in the pipeline'
            self.__outputs = frozenset(outputs)
        else:
            self.__outputs = None

    def _map_input_names_to_slots(self, input_values: Mapping[str, Data]) -> Dict[BlockInputSlot, Data]:
        """Method to translate dictionary, passed in :meth:`__call__`, to dictionary that is useful for evaluation.
        Args:
            input_values: Input data, passed to the :meth:`__call__` method.
        Returns:
            Remapped input data.
        """
        inp_slots_to_data_map = dict()
        for name, data in input_values.items():
            input_slot = self.__pipeline.inputs.get(name, None)
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
            if self.__inputs is not None:
                assert name in self.__inputs, f'Input "{name}" is not in the executor\'s inputs set'
            input_slot = self.__pipeline.inputs.get(name, None)
            if input_slot is None:
                warnings.warn(f'Unable to find input "{name}" in the pipeline')

    def _is_slot_required(self, slot: BaseSlot) -> bool:
        """Method that determines if the slot is required during
        the computational graph execution. Added for additional debugging (and polymorphism)
        features and to make the code shorter.

        Args:
            slot: The computational block's slot to check.
        """
        return self.__slot_handler.is_slot_required(self.stage, slot)

    def _execute_block(self, block: BaseBlock, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Method that executes the block. Added for additional debugging (and polymorphism)
        features and to make the code shorter.

        Args:
            block: The computational block to execute.
            block_input_mapping: The data for the block execution.
        """
        return self.__block_executor.execute_block(self.stage, block, block_input_mapping)

    def __call__(self, input_values: Mapping[str, Union[BaseData, np.ndarray]]) -> ExecutionResult:
        """Executes the pipeline given `BaseData` or just NumPy arrays and returns results dictionary.

        Args:
            input_values: Input data.

        Returns:
            Calculated output data.
        """
        input_values_data = {
            name: data if isinstance(data, BaseData) else CPUData(data)
            for name, data in input_values.items()
        }
        outputs = self.execute(input_values_data)
        return ExecutionResult(outputs)

    @abstractmethod
    def execute(self, input_values: Mapping[str, BaseData]) -> Dict[str, BaseData]:
        """Executes the pipeline given `BaseData` inputs and return `BaseData` output values.

        Args:
            input_values: Input data.

        Returns:
            Calculated output data dictionary that maps output names to the data.
        """
        ...

    @property
    def pipeline(self) -> BasePipeline:
        """Getter for the executor's pipeline."""
        return self.__pipeline

    @property
    def inputs(self) -> Optional[FrozenSet[str]]:
        """Getter for the executor's inputs set. `None` if there are
        no restrictions on the pipeline's inputs."""
        return self.__inputs

    @property
    def outputs(self) -> Optional[FrozenSet[str]]:
        """Getter for the executor's ouputs set. `None` if there are
        no restrictions on the pipeline's outputs."""
        return self.__outputs

    @property
    def slot_handler(self) -> BaseSlotHandler:
        """Getter for the executor's slots handler."""
        return self.__slot_handler

    @property
    def block_executor(self) -> BaseBlockExecutor:
        """Getter for the executor's block executor."""
        return self.__block_executor

    @property
    def stage(self) -> Stage:
        """Getter for the executor's computational stage."""
        return self.__stage

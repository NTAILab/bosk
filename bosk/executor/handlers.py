from abc import ABC, abstractmethod
from typing import Mapping

# from .base import InputSlotToDataMapping
from ..data import Data
from ..stages import Stage
from ..block.slot import BlockInputSlot, BaseSlot
from ..block.base import BaseBlock, BlockOutputData

InputSlotToDataMapping = Mapping[BlockInputSlot, Data]  # circular import


class BaseHandler(ABC):
    """The interface for the classes, parametrizing executor's behaviour
    during the execution process.

    Attributes:
        __stage: The computational stage performed by the handler.

    Args:
        stage: The computational stage performed by the handler.
    """

    __stage: Stage

    def __init__(self, stage) -> None:
        self.__stage = stage

    @property
    def stage(self) -> Stage:
        """Getter for the handler's computational stage."""
        return self.__stage


class BaseSlotHandler(BaseHandler):
    """Determines slots' handling policy.

    Args:
        stage: The computational stage performed by the handler.
    """

    def __init__(self, stage) -> None:
        super().__init__(stage)

    @abstractmethod
    def is_slot_required(self, slot: BaseSlot) -> bool:
        """Method that determines if the slot is required during
        the computational graph execution.

        Args:
            slot: The computational block's slot to check.
        """


class BaseBlockHandler(BaseHandler):
    """Determines blocks' handling policy.

    Args:
        stage: The computational stage performed by the handler.
    """

    def __init__(self, stage) -> None:
        super().__init__(stage)

    @abstractmethod
    def execute_block(self, block: BaseBlock, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Method that executes the block.

        Args:
            block: The computational block to execute.
            block_input_mapping: The data for the block execution.
        """


class DefaultSlotHandler(BaseSlotHandler):
    def __init__(self, stage: Stage) -> None:
        assert (stage == Stage.FIT or stage == Stage.TRANSFORM), "Stage is not implemented"
        super().__init__(stage)

    def is_slot_required(self, slot: BaseSlot) -> bool:
        assert isinstance(slot, BlockInputSlot), "InputSlotStrategy proceeds only input slots"
        if self.stage == Stage.FIT:
            return slot.meta.stages.fit \
                or slot.meta.stages.transform \
                or slot.meta.stages.transform_on_fit
        return slot.meta.stages.transform


class DefaultBlockHandler(BaseBlockHandler):
    def __init__(self, stage: Stage) -> None:
        assert (stage == Stage.FIT or stage == Stage.TRANSFORM), "Stage is not implemented"
        super().__init__(stage)

    def execute_block(self, block: BaseBlock, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if self.stage == Stage.FIT:
            block.fit({
                slot.meta.name: values
                for slot, values in block_input_mapping.items()
                if slot.meta.stages.fit
            })
        filtered_block_input_mapping = {
            slot.meta.name: values
            for slot, values in block_input_mapping.items()
            if slot.meta.stages.transform or (self.stage == Stage.FIT and slot.meta.stages.transform_on_fit)
        }
        return block.wrap(block.transform(filtered_block_input_mapping))

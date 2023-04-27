from abc import ABC, abstractmethod
from typing import Mapping, Optional, Sequence

from ..data import Data
from ..stages import Stage
from ..block.base import BaseBlock, BlockOutputData, BlockGroup, BlockInputSlot


InputSlotToDataMapping = Mapping[BlockInputSlot, Data]
"""Block input slot data mapping.

It is indexed by input slots.
"""


class BaseBlockExecutor(ABC):
    """Determines a block execution.

    """

    @abstractmethod
    def execute_block(self, stage: Stage, block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Method that executes the block.

        Args:
            stage: The execution stage.
            block: The computational block to execute.
            block_input_mapping: The data for the block execution.
        """


class DefaultBlockExecutor(BaseBlockExecutor):
    def execute_block(self, stage: Stage,
                      block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if stage == Stage.FIT:
            block.fit({
                slot.meta.name: values
                for slot, values in block_input_mapping.items()
                if slot.meta.stages.fit
            })
        filtered_block_input_mapping = {
            slot.meta.name: values
            for slot, values in block_input_mapping.items()
            if slot.meta.stages.transform or (stage == Stage.FIT and slot.meta.stages.transform_on_fit)
        }
        return block.wrap(block.transform(filtered_block_input_mapping))


class FitBlacklistBlockExecutor(DefaultBlockExecutor):
    """Block executor that does not call `fit` method for the blocks that are in the black list.

    It is used to be able to manually fit some blocks of the pipeline and avoid overriding of the state
    when the whole pipeline is fitted.
    """

    def __init__(self, ignore_blocks: Optional[Sequence[BaseBlock]] = None,
                 ignore_groups: Optional[Sequence[BlockGroup]] = None) -> None:
        """Initialize the block executor.

        Args:
            ignore_blocks: List of blocks to ignore.
            ignore_groups: List of block groups to ignore.
                           If block belongs to at least one group from `ignore_groups`,
                           it won't be fitted.

        """
        super().__init__()
        self.ignore_blocks = set(ignore_blocks or [])
        self.ignore_groups = set(ignore_groups or [])

    def execute_block(self, stage: Stage, block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        need_ignore_fit = (block in self.ignore_blocks) or (not block.slots.groups.isdisjoint(self.ignore_groups))
        if need_ignore_fit:
            # avoid block fitting, just apply transform
            filtered_block_input_mapping = {
                slot.meta.name: values
                for slot, values in block_input_mapping.items()
                if slot.meta.stages.transform or (stage == Stage.FIT and slot.meta.stages.transform_on_fit)
            }
            return block.wrap(block.transform(filtered_block_input_mapping))
        return super().execute_block(stage, block, block_input_mapping)

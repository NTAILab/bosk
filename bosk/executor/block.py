"""Block execution module.

When a pipeline is executed, block executors, derived from the :py:class:`BaseBlockExecutor`
are used to evaluate result of each block.

"""

from abc import ABC, abstractmethod
from typing import Mapping, MutableMapping, Optional, Sequence

from ..data import BaseData, CPUData, GPUData
from ..stages import Stage
from ..block.base import BaseBlock, BlockOutputData, BlockGroup, BlockInputSlot


__all__ = [
    "BaseBlockExecutor",
    "DefaultBlockExecutor",
    "GPUBlockExecutor",
    "CPUBlockExecutor",
    "FitBlacklistBlockExecutor",
]


InputSlotToDataMapping = Mapping[BlockInputSlot, BaseData]
"""Block input slot data mapping.

It is indexed by input slots.
"""


class BaseBlockExecutor(ABC):
    """Determines a block execution.

    """

    @abstractmethod
    def execute_block(self, stage: Stage, block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Execute the `block` at the `stage` given `block_input_mapping` data dictionary.

        Args:
            stage: The execution stage.
            block: The computational block to execute.
            block_input_mapping: The data for the block execution.

        """


class DefaultBlockExecutor(BaseBlockExecutor):
    """Default block executor.

    Prepares arguments and calls `transform` method at both `FIT` and `TRANSFORM` stages,
    and before that `fit` method at `FIT` stage.

    """
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


class GPUBlockExecutor(BaseBlockExecutor):
    """Block executor that tries to execute block on GPU, if it is possible.

    """
    def execute_block(self, stage: Stage,
                      block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if stage == Stage.FIT:
            filtered_block_input_mapping_fit: MutableMapping[str, BaseData] = dict()
            for slot, values in block_input_mapping.items():
                if slot.meta.stages.fit:
                    if slot.parent_block.meta.execution_props.gpu:
                        if isinstance(values, BaseData) or isinstance(values, CPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values.to_gpu()
                        elif isinstance(values, GPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values
                        else:
                            filtered_block_input_mapping_fit[slot.meta.name] = GPUData(values)
                    else:
                        if isinstance(values, BaseData) or isinstance(values, GPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values.to_cpu()
                        elif isinstance(values, CPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values
                        else:
                            filtered_block_input_mapping_fit[slot.meta.name] = CPUData(values)
            block.fit(filtered_block_input_mapping_fit)

        filtered_block_input_mapping: MutableMapping[str, BaseData] = dict()
        for slot, values in block_input_mapping.items():
            if slot.meta.stages.transform or (stage == Stage.FIT and slot.meta.stages.transform_on_fit):
                if slot.parent_block.meta.execution_props.gpu:
                    if isinstance(values, BaseData) or isinstance(values, CPUData):
                        filtered_block_input_mapping[slot.meta.name] = values.to_gpu()
                    elif isinstance(values, GPUData):
                        filtered_block_input_mapping[slot.meta.name] = values
                    else:
                        filtered_block_input_mapping[slot.meta.name] = GPUData(values)
                else:
                    if isinstance(values, BaseData) or isinstance(values, GPUData):
                        filtered_block_input_mapping[slot.meta.name] = values.to_cpu()
                    elif isinstance(values, CPUData):
                        filtered_block_input_mapping[slot.meta.name] = values
                    else:
                        filtered_block_input_mapping[slot.meta.name] = CPUData(values)

        output = block.transform(filtered_block_input_mapping)

        return block.wrap(output)


class CPUBlockExecutor(BaseBlockExecutor):
    """Block executor that tries to execute block on CPU, if it is possible.

    """
    def execute_block(self, stage: Stage,
                      block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if stage == Stage.FIT:
            filtered_block_input_mapping_fit: MutableMapping[str, BaseData] = dict()
            for slot, values in block_input_mapping.items():
                if slot.meta.stages.fit:
                    if slot.parent_block.meta.execution_props.cpu:
                        if isinstance(values, BaseData) or isinstance(values, GPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values.to_cpu()
                        elif isinstance(values, CPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values
                        else:
                            filtered_block_input_mapping_fit[slot.meta.name] = CPUData(values)
                    else:
                        if isinstance(values, BaseData) or isinstance(values, CPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values.to_gpu()
                        elif isinstance(values, GPUData):
                            filtered_block_input_mapping_fit[slot.meta.name] = values
                        else:
                            filtered_block_input_mapping_fit[slot.meta.name] = GPUData(values)
            block.fit(filtered_block_input_mapping_fit)

        filtered_block_input_mapping: MutableMapping[str, BaseData] = dict()
        for slot, values in block_input_mapping.items():
            if slot.meta.stages.transform or (stage == Stage.FIT and slot.meta.stages.transform_on_fit):
                if slot.parent_block.meta.execution_props.cpu:
                    if isinstance(values, BaseData) or isinstance(values, GPUData):
                        filtered_block_input_mapping[slot.meta.name] = values.to_cpu()
                    elif isinstance(values, CPUData):
                        filtered_block_input_mapping[slot.meta.name] = values
                    else:
                        filtered_block_input_mapping[slot.meta.name] = CPUData(values)
                else:
                    if isinstance(values, BaseData) or isinstance(values, CPUData):
                        filtered_block_input_mapping[slot.meta.name] = values.to_gpu()
                    elif isinstance(values, GPUData):
                        filtered_block_input_mapping[slot.meta.name] = values
                    else:
                        filtered_block_input_mapping[slot.meta.name] = GPUData(values)

        return block.wrap(block.transform(filtered_block_input_mapping))


class FitBlacklistBlockExecutor(DefaultBlockExecutor):
    """Block executor that does not call `fit` method for the blocks that are in the black list.

    It is used to be able to manually fit some blocks of the pipeline and avoid overriding of the state
    when the whole pipeline is fitted.

    Args:
        ignore_blocks: List of blocks to ignore.
        ignore_groups: List of block groups to ignore.
                        If block belongs to at least one group from `ignore_groups`,
                        it won't be fitted.

    """

    def __init__(self, ignore_blocks: Optional[Sequence[BaseBlock]] = None,
                 ignore_groups: Optional[Sequence[BlockGroup]] = None) -> None:
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

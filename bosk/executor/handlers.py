from .base import BaseSlotStrategy, BaseExecutionStrategy, InputSlotToDataMapping
from ..stages import Stage
from ..slot import BlockInputSlot, BaseSlot
from ..block.base import BaseBlock, BlockOutputData

class InputSlotStrategy(BaseSlotStrategy):
    def __init__(self, stage: Stage) -> None:
        assert(stage == Stage.FIT or stage == Stage.TRANSFORM), "Stage is not implemented"
        self.stage = stage

    def is_slot_required(self, slot: BaseSlot) -> bool:
        assert isinstance(slot, BlockInputSlot), "InputSlotStrategy proceeds only input slots"
        if self.stage == Stage.FIT:
            return slot.meta.stages.fit \
                or slot.meta.stages.transform \
                or slot.meta.stages.transform_on_fit
        return slot.meta.stages.transform

class SimpleExecutionStrategy(BaseExecutionStrategy):
    def __init__(self, stage: Stage) -> None:
        assert(stage == Stage.FIT or stage == Stage.TRANSFORM), "Stage is not implemented"
        self.stage = stage

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
from .base import BaseBlockExecutor, BaseBlock, InputSlotToDataMapping
from ..block.base import BlockOutputData
from ..block.zoo.input_plugs import TargetInputBlock
from ..stages import Stage
from ..utility import timer_wrap
from typing import Dict


class TimerBlockHandler(BaseBlockExecutor):
    def __init__(self) -> None:
        super().__init__()
        self._time_dict: Dict[BaseBlock, float] = dict()

    def execute_block(self, stage: Stage,
                      block: BaseBlock,
                      block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        fit_time = 0.0
        if stage == Stage.FIT:
            _, fit_time = timer_wrap(block.fit)({
                slot.meta.name: values
                for slot, values in block_input_mapping.items()
                if slot.meta.stages.fit
            })
        filtered_block_input_mapping = {
            slot.meta.name: values
            for slot, values in block_input_mapping.items()
            if slot.meta.stages.transform or (stage == Stage.FIT and slot.meta.stages.transform_on_fit)
        }
        tf_res, tf_time = timer_wrap(block.transform)(filtered_block_input_mapping)
        self._time_dict[block] = fit_time + tf_time
        return block.wrap(tf_res)

    @property
    def blocks_time(self) -> Dict[BaseBlock, float]:
        return self._time_dict.copy()

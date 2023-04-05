from typing import Optional
from ...base import BaseOutputBlock, BlockInputData, TransformOutputData
from ...meta import BlockExecutionProperties, make_simple_meta


class OutputBlock(BaseOutputBlock):
    DEFAULT_OUTPUT_NAME = 'out'
    name = None
    meta = None

    def __init__(self, name: Optional[str] = None):
        slot_name = name if name is not None else self.DEFAULT_OUTPUT_NAME
        self.meta = make_simple_meta(
            [slot_name],
            [slot_name],
            execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True)
        )
        super().__init__()
        self.name = name

    def fit(self, _inputs: BlockInputData) -> 'OutputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs

from typing import Optional
from ...base import BaseInputBlock, BlockInputData, TransformOutputData
from ....stages import Stages
from ...meta import BlockMeta, BlockExecutionProperties, DynamicBlockMetaStub, make_simple_meta


class InputBlock(BaseInputBlock):
    DEFAULT_INPUT_NAME = 'X'
    name: Optional[str]
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, name: Optional[str] = None):
        slot_name = name if name is not None else self.DEFAULT_INPUT_NAME
        self.meta = make_simple_meta(
            [slot_name],
            [slot_name],
            execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True)
        )
        super().__init__()
        self.name = name

    def fit(self, _inputs: BlockInputData) -> 'InputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs


class TargetInputBlock(BaseInputBlock):
    DEFAULT_TARGET_NAME = 'y'
    name: Optional[str]
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, name: Optional[str] = None):
        slot_name = name if name is not None else self.DEFAULT_TARGET_NAME
        self.meta = make_simple_meta(
            [slot_name],
            [slot_name],
            execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True)
        )
        super().__init__()
        self.name = name

    def fit(self, _inputs: BlockInputData) -> 'TargetInputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs

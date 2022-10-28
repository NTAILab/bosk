from bosk.block import BaseBlock, BlockInputData, TransformOutputData
from bosk.stages import Stages
from bosk.block import BlockMeta
from bosk.slot import InputSlotMeta, OutputSlotMeta
from bosk.block.meta import make_simple_meta


class InputBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['X'])

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'InputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs


class TargetInputBlock(BaseBlock):
    TARGET_NAME = 'y'

    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name=TARGET_NAME,
                stages=Stages(transform=False, transform_on_fit=True),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name=TARGET_NAME,
            )
        ]
    )

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'TargetInputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs
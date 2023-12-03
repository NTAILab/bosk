from typing import Optional
from ...base import BaseInputBlock, BlockInputData, TransformOutputData
from ...placeholder import PlaceholderMixin
from ....stages import Stages
from ...meta import BlockMeta, BlockExecutionProperties, DynamicBlockMetaStub, make_simple_meta


__all__ = [
    "Input",
    "TargetInput",
    # for backward compatibility:
    "InputBlock",
    "TargetInputBlock",
]


class Input(PlaceholderMixin, BaseInputBlock):
    """Input block.

    Bypasses the input. Can be used to make pipeline in functional style.

    Dynamically specifies the meta information.

    Args:
        name: The input slot name. If None, the default name is used.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `name` (default="X"): Feature data array.

    Output slots
    ------------

        - `name` (default="X"): Feature data array.

    """
    DEFAULT_INPUT_NAME = 'X'
    name: Optional[str] = None
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
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs


class TargetInput(PlaceholderMixin, BaseInputBlock):
    """Target input block.

    Bypasses the input. Can be used to make pipeline in a functional style.

    Dynamically specifies the meta information.

    Args:
        name: The target input slot name. If None, the default name is used.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `name` (default="y"): Target data array.

    Output slots
    ------------

        - `name` (default="y"): Target data array.

    """
    DEFAULT_TARGET_NAME = 'y'
    name: Optional[str] = None
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
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return inputs


InputBlock = Input
TargetInputBlock = TargetInput


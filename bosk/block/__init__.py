"""Blocks are an essential part of a pipeline (:py:class:`bosk.pipeline.BasePipeline`).

Any block can be fitted on some data and then used to transform the data.

Blocks may have different number of inputs and outputs.
Information about inputs, outputs and other block properties
is stored in :py:class:`bosk.block.meta.BlockMeta`.
The meta information may be shared between different instances of the same block class.
Each input is described by a :py:class:`bosk.block.meta.InputSlotMeta` and
each output is described by a :py:class:`bosk.block.meta.OutputSlotMeta`.

To connect blocks to each other, every block has `slots`:
:py:class:`BlockInputSlot` and :py:class:`BlockOutputSlot`,
which correspond to meta information. But the slots are unique for each block (i.e. block class instance).

Every block class should be derived from :py:class:`bosk.block.base.BaseBlock`,
define meta information and implement the :py:meth:`fit` and :py:meth:`transform` methods.

*Note*, that block prediction is always performed with the :py:meth:`transform` method, not with "predict"
or something else.

List of available blocks can be found in the :py:mod:`bosk.block.zoo`.

"""
from .auto import auto_block
from .base import (
    BaseBlock,
    BlockInputData,
    BlockOutputData,
    TransformOutputData,
    BaseInputBlock,
    BaseOutputBlock,
    BaseSlot,
    BlockInputSlot,
    BlockOutputSlot,
    SlotT,
    BaseSlotMeta,
    BlockGroup
)
from .meta import BlockMeta, InputSlotMeta, OutputSlotMeta

__all__ = [
    "auto_block",
    "BlockMeta",
    "InputSlotMeta",
    "OutputSlotMeta",
    "BlockInputData",
    "TransformOutputData",
    "BlockOutputData",
    "BaseBlock",
    "BaseInputBlock",
    "BaseOutputBlock",
    "BaseSlot",
    "BlockInputSlot",
    "BlockOutputSlot",
    "SlotT",
    "BlockGroup",
    "BaseSlotMeta",
]

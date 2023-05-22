import numpy as np
from typing import Optional, Union
from sklearn.model_selection import StratifiedKFold

from ...base import BaseBlock
from ....stages import Stages
from ....data import CPUData
from ...meta import BlockMeta, BlockExecutionProperties, DynamicBlockMetaStub, InputSlotMeta, OutputSlotMeta


class CVTrainIndicesBlock(BaseBlock):
    """Cross-validation Training Indices Block.

    Generates training indices for `size` models.
    The block has `size` outputs, each named as a number of model: `"0", "1", ...`.

    Dynamically specifies the meta information.

    Args:
        size: Cross-validation size (number of splits).
        random_state: Random state.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Features data array.
        - y: Target variable array.

    Output slots
    ------------

        - "0": Fold 0 training subset indices.
        - ...
        - "<n>": Fold n training subset indices.

    Attributes:
        size: Cross-validation size (number of splits).
        random_state: Random state.

    """

    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, size: int, random_state: Optional[int]):
        self.size = size
        self.random_state = random_state
        self.meta = BlockMeta(
            inputs=[
                InputSlotMeta(name=name, stages=Stages(fit=True, transform=False, transform_on_fit=True))
                for name in ('X', 'y')
            ],
            outputs=[
                OutputSlotMeta(name=str(i))
                for i in range(size)
            ],
            execution_props=BlockExecutionProperties(plain=True)
        )
        super().__init__()

    def fit(self, inputs):
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs):
        X = inputs['X'].data
        y = inputs['y'].data
        kfold = StratifiedKFold(n_splits=self.size, shuffle=True, random_state=self.random_state)
        return {
            str(i): CPUData(train_idx)
            for i, (train_idx, _test_idx) in enumerate(kfold.split(X, y))
        }


class SubsetTrainWrapperBlock(BaseBlock):
    """Block wrapper that fits the base block on the indices subset.

    The base block may have an arbitrary number of inputs of any type,
    but should not accept input named `"trainin_indices"`.

    At FIT stage the wrapper extracts subsets of each input along the first dimension.

    At TRANSFORM stage the wrapper bypasses inputs to the base block.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - All inputs from the underlying block `block`.
        - training_indices: Training indices.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - All inputs from the underlying block `block`.

    Output slots
    ------------

        - All outputs from the underlying block `block`.

    Attributes:
        block: Underlying block.

    """
    TRAINING_INDICES_NAME = 'training_indices'

    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, block: BaseBlock):
        """Initialize the subset training wrapper block.

        Dynamically specifies the meta information.

        The meta information is determined by the underlying block `block`.

        Args:
            block: The underlying block.

        """
        self.meta = BlockMeta(
            inputs=[
                inp for inp in block.meta.inputs.values()
            ] + [
                InputSlotMeta(
                    name='training_indices',
                    stages=Stages(fit=True, transform=False, transform_on_fit=True)
                )
            ],
            outputs=[
                out for out in block.meta.outputs.values()
            ],
            execution_props=block.meta.execution_props
        )
        self.block = block
        super().__init__()

    def _exclude_training_indices(self, inputs):
        return {
            k: v
            for k, v in inputs.items()
            if k != self.TRAINING_INDICES_NAME
        }

    def fit(self, inputs):
        ids = inputs[self.TRAINING_INDICES_NAME].data
        block_inputs = {
            k: v.__class__(v.data[ids])
            for k, v in self._exclude_training_indices(inputs).items()
        }
        self.block.fit(block_inputs)
        return self

    def transform(self, inputs):
        block_inputs = self._exclude_training_indices(inputs)
        return self.block.transform(block_inputs)

    def set_random_state(self, seed: Optional[Union[int, np.random.Generator]]) -> None:
        return self.block.set_random_state(seed)

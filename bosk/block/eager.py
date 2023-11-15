from ..executor.block import BaseBlockExecutor, InputSlotToDataMapping
from ..block.base import BaseBlock, BlockOutputData
from ..data import BaseData
from ..stages import Stage
from .functional import FunctionalBlockWrapper
from typing import Optional, Mapping


class EagerBlockState:
    """Eager block state, containing the results of block execution (output data).

    The block state is strongly related to the block, but not to its wrapper.
    I.e. it can be shared between different block wrappers corresponding to the same block.

    Attributes:
        fit_output_values: Eagerly calculated output values of the block.
                           When `None`, the block has not yet been executed.

    """
    def __init__(self):
        self.fit_output_values: Optional[Mapping[str, BaseData]] = None


class EagerBlockWrapper(FunctionalBlockWrapper):
    """Block wrapper that contains the evaluation state (its output data).

    It can be used for eager evaluations at pipeline construction step.
    Comparing to a regular `FunctionalBlockWrapper`, it can be executed by `execute(...)`
    and return calculated output data by `get_output_data()`.

    Attributes:
        executor: Block executor.
        state: Eager block state.

    Args:
        block: Block to be wrapped.
        executor: Block executor that will be applied to th block to update the state.
        output_name: Output slot name (the same as in `FunctionalBlockWrapper`),
                     is used to specify which output is needed.

    """
    def __init__(self, block: BaseBlock, executor: BaseBlockExecutor,
                 output_name: Optional[str] = None):
        super().__init__(block, output_name=output_name)
        self.executor = executor
        self.state = EagerBlockState()

    def execute(self, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Execute the underlying block with the given inputs and store the result in the state.

        The result stored in `fit_output_values` and can be accessed by `get_output_data()`.

        Args:
            inputs: Block inputs.

        """
        assert self.state.fit_output_values is None, 'Cannot fit the eager block twice'
        block_output_data = self.executor.execute_block(
            stage=Stage.FIT,
            block=self.block,
            block_input_mapping=block_input_mapping,
        )
        self.state.fit_output_values = block_output_data
        return block_output_data

    def get_output_data(self) -> BlockOutputData:
        """Get the output data from the state for the current output slot.

        Use `[...]` operator to obtain block wrapper with the same state, but different
        selected output.

        Returns:
            Output data for the current output.

        """
        assert self.state.fit_output_values is not None
        return self.state.fit_output_values[self.get_output_slot()]

    def __getitem__(self, output_name: str) -> 'EagerBlockWrapper':
        """Get the eager block wrapper of the same block for the different output name.

        Args:
            output_name: Name of output to select.

        Returns:
            New block wrapper for the block with the `output_name` output.

        """
        eager_block = EagerBlockWrapper(self.block, self.executor, output_name=output_name)
        eager_block.state = self.state
        return eager_block


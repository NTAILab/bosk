from ...executor.block import BaseBlockExecutor, InputSlotToDataMapping, DefaultBlockExecutor
from ...block.eager import EagerBlockWrapper
from ...block.base import BaseBlock, BaseInputBlock
from ...block.zoo.routing.shared import SharedProducer, SharedConsumer
from ...data import BaseData, CPUData
from .functional import FunctionalPipelineBuilder, BaseBlockClassRepository, Connection, PlaceholderFunctionState
from ...exceptions import BlockReuseError
from typing import Optional, Callable
import numpy as np
import inspect


class EagerPipelineBuilder(FunctionalPipelineBuilder):
    """Pipeline builder with eager evaluations.

    The builder executes each block at the moment when inputs are passed to its placeholder.

    Args:
        block_executor: Block executor, which will be applied to blocks to calculate outputs.
        block_repo: Block class repository for resolving blocks by their names.
                    Default is zoo scope repository, which means that
                    all blocks defined in :py:mod:`bosk.block.zoo` will be available
                    (without postfix "Block", for example:
                    :py:class:`bosk.block.zoo.data_conversion.ArgmaxBlock`
                    should be accessed as just "Argmax").

    """
    def __init__(self, block_executor: Optional[BaseBlockExecutor] = None,
                 block_repo: Optional[BaseBlockClassRepository] = None,
                 allow_block_reuse: bool = True):
        super().__init__(block_repo, allow_block_reuse=allow_block_reuse)
        if block_executor is None:
            block_executor = DefaultBlockExecutor()
        self.block_executor = block_executor

    def _make_placeholder_fn(self, block: BaseBlock,
                             state: Optional[PlaceholderFunctionState] = None) -> Callable[..., EagerBlockWrapper]:
        """Make a placeholder function for the block.

        Args:
            block: Block for which a placeholder is needed.

        Returns:
            Placeholder function that can be applied to other block wrappers or to data.
            This function will execute the block.

        """
        if state is None:
            state = PlaceholderFunctionState(block)

        def placeholder_fn(*pfn_args, **pfn_kwargs) -> EagerBlockWrapper:
            """Execute the block with the given arguments.

            Changes the state of the underlying block (applies FIT & TRANSFORM).

            Args:
                pfn_args: Only one unnamed argument is supported if the block is `BaseInputBlock`.
                pfn_kwargs: Named arguments can be other blocks wrappers or just data (`BaseData`).

            Returns:
                Block wrapper.

            """
            block, producer_block = self._handle_block_reuse(state)

            if len(pfn_args) > 0:
                assert len(pfn_kwargs) == 0, \
                    'Either unnamed or named arguments can be used, but not at the same time'
                assert len(pfn_args) == 1, \
                    'Only one unnamed argument is supported (we can infer name only in this case)'
                # assert isinstance(block, BaseInputBlock), 'Only input blocks support unnamed arguments'
                pfn_kwargs = {
                    block.get_single_input().meta.name: pfn_args[0]
                }
            block_input_mapping: InputSlotToDataMapping = dict()
            for input_name, input_block_wrap_or_data in pfn_kwargs.items():
                block_input = block.slots.inputs[input_name]
                if isinstance(input_block_wrap_or_data, EagerBlockWrapper):
                    self._connections.append(
                        Connection(
                            src=input_block_wrap_or_data.get_output_slot(),
                            dst=block_input,
                        )
                    )
                    block_input_mapping[block_input] = input_block_wrap_or_data.get_output_data()
                elif isinstance(input_block_wrap_or_data, BaseData):
                    block_input_mapping[block_input] = input_block_wrap_or_data
                elif isinstance(input_block_wrap_or_data, np.ndarray):
                    block_input_mapping[block_input] = CPUData(input_block_wrap_or_data)
                else:
                    raise ValueError(
                        f'Wrong placeholder input type: {type(input_block_wrap_or_data)}'
                    )
            if isinstance(block, SharedConsumer):
                assert producer_block is not None
                block_input_mapping[block.slots.inputs[block.input_block_name]] = CPUData(
                    np.array(producer_block.block)
                )
            eager_block = EagerBlockWrapper(block, executor=self.block_executor)
            if len(block_input_mapping) > 0:
                eager_block.execute(block_input_mapping)
            self._register_block_wrapper(block, eager_block)
            return eager_block

        # make appropriate docstring and function signature
        placeholder_fn.__doc__ = f'Execute the block {block} and get the result wrapper.' \
            '\n\n' + ('=' * (25 + len(repr(block)))) + '\n' \
            f'The block {block} documentation:\n\n' + str(inspect.getdoc(block.__class__))
        setattr(
            placeholder_fn,
            '__signature__',
            inspect.Signature([
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                for name in block.slots.inputs.keys()
            ])
        )
        return placeholder_fn


from ...executor.block import BaseBlockExecutor, InputSlotToDataMapping
from ...block.eager import EagerBlockWrapper
from ...block.base import BaseBlock, BaseInputBlock
from ...data import BaseData
from .functional import FunctionalPipelineBuilder, BaseBlockClassRepository, Connection
from typing import Optional, Callable


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
    def __init__(self, block_executor: BaseBlockExecutor,
                 block_repo: Optional[BaseBlockClassRepository] = None):
        super().__init__(block_repo)
        self.block_executor = block_executor

    def _make_placeholder_fn(self, block: BaseBlock) -> Callable[..., EagerBlockWrapper]:
        """Make a placeholder function for the block.

        Args:
            block: Block for which a placeholder is needed.

        Returns:
            Placeholder function that can be applied to other block wrappers or to data.
            This function will execute the block.

        """
        def placeholder_fn(*pfn_args, **pfn_kwargs) -> EagerBlockWrapper:
            """Execute the block with the given arguments.

            Changes the state of the underlying block (applies FIT & TRANSFORM).

            Args:
                pfn_args: Only one unnamed argument is supported if the block is `BaseInputBlock`.
                pfn_kwargs: Named arguments can be other blocks wrappers or just data (`BaseData`).

            Returns:
                Block wrapper.

            """
            if len(pfn_args) > 0:
                assert len(pfn_kwargs) == 0, \
                    'Either unnamed or named arguments can be used, but not at the same time'
                assert len(pfn_args) == 1, \
                    'Only one unnamed argument is supported (we can infer name only in this case)'
                assert isinstance(block, BaseInputBlock)
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
                else:
                    raise ValueError(
                        f'Wrong placeholder input type: {type(input_block_wrap_or_data)}'
                    )
            eager_block = EagerBlockWrapper(block, executor=self.block_executor)
            if len(block_input_mapping) > 0:
                eager_block.execute(block_input_mapping)
            return eager_block

        return placeholder_fn


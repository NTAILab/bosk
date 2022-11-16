from typing import Mapping, Union
from ...block import BaseBlock
from ...block.functional import FunctionalBlockWrapper
from ...block.repo import BaseBlockClassRepository, DEFAULT_BLOCK_CLASS_REPOSITORY
from ...slot import BlockInputSlot, BlockOutputSlot
from ..connection import Connection
from ..base import BasePipeline
from .base import BasePipelineBuilder
from typing import List, Optional, Callable


class FunctionalPipelineBuilder(BasePipelineBuilder):
    """Pipeline builder with functional interface.
    """
    def __init__(self, block_repo: Optional[BaseBlockClassRepository] = None):
        """Initialize functional pipeline builder.

        Args:
            block_repo: Block class repository for resolving blocks by their names.

        """
        self._nodes: List[BaseBlock] = []
        self._connections: List[Connection] = []
        if block_repo is None:
            block_repo = DEFAULT_BLOCK_CLASS_REPOSITORY
        self._block_repo: BaseBlockClassRepository = block_repo

    def __getattr__(self, name: str) -> Callable:
        block_cls = self._block_repo.get(name)
        return self._get_block_init(block_cls)

    def _register_block(self, block: BaseBlock):
        """Register block in the builder.

        Args:
            block: Block to register.

        """
        self._nodes.append(block)

    def _make_placeholder_fn(self, block: BaseBlock) -> Callable:
        def placeholder_fn(*pfn_args, **pfn_kwargs):
            """Placeholder function.

            Placeholder function operates with functional block wrappers:
            it takes them as inputs and returns one as output.

            The main reason to use placeholder function is that it
            connects output slots of input wrappers with input slots
            of the underlying block.

            Args:
                *pfn_args: Not supported.
                *pfn_kwargs: Inputs functional wrappers.
            """
            assert len(pfn_args) == 0, "Only kwargs are supported"
            for input_name, input_block_wrapper in pfn_kwargs.items():
                self._connections.append(
                    Connection(
                        src=input_block_wrapper.get_output_slot(),
                        dst=block.slots.inputs[input_name],
                    )
                )
            return FunctionalBlockWrapper(block)

        return placeholder_fn

    def wrap(self, block: BaseBlock) -> Callable:
        """Register the block in the builder and wrap it into a placeholder function.

        Args:
            block: Block to wrap.

        Returns:
            Placeholder function.

        Examples:
            Assume some block `test_block` was created before builder initialization.
            If we want to add the block into the pipeline,
            it should be wrapped:

            >>> test_block = RFCBlock()  # Random Forest Classifier
            >>> b = FunctionalPipelineBuilder()
            >>> rf = b.wrap(test_block)  # register the block in the builder
            >>> x = b.Input()
            >>> result = rf(X=x)

        """
        self._register_block(block)
        return self._make_placeholder_fn(block)

    def _get_block_init(self, block_cls: Callable) -> Callable:
        """Get a new block initialization wrapper.

        Args:
            block_cls: Block class.

        Returns:
            Block initialization wrapper.
            It takes arguments for the block class, adds the block
            to the pipeline and returns a placeholder function.
            The placeholder function takes functional block wrappers
            as inputs and returns functional block wrapper.

        """
        def block_init(*args, **kwargs):
            """Initialize block class with the given arguments and
            return a placeholder function.

            Args:
                *args: Arguments for block initialization.
                *kwargs: Keyword arguments for block initialization.

            Returns:
                Placeholder function for the constructed block.

            """
            block = block_cls(*args, **kwargs)
            return self.wrap(block)

        return block_init

    def new(self, block_cls: Callable, *args, **kwargs) -> Callable:
        """Make a new block wrapper of given block class.

        Constructs block wrapper using given block class constructor
        and provided arguments.
        Can be used for custom block classes that can't be found by
        block class repository (see `__init__`).

        Args:
            block_cls: Block class.
            *args: Arguments for the block constructor.
            **kwargs: Keyword arguments for the block constructor.

        Returns:
            Placeholder function which will get constructed block arguments.

        """
        return self._get_block_init(block_cls)(*args, **kwargs)

    def build_pipeline(self, inputs: Mapping[str, Union[BlockInputSlot, FunctionalBlockWrapper]],
                        outputs: Mapping[str, Union[BlockOutputSlot, FunctionalBlockWrapper]]) -> BasePipeline:
        """Build and get pipeline.

        Args:
            inputs: Dictionary containing the information about pipeline's inputs. See :arrt:`BasePipeline.inputs`.
            outputs: Dictionary containing the information about pipeline's outputs. See :arrt:`BasePipeline.outputs`

        Returns:
            Pipeline made from wrapped blocks.

        """
        inp_dict = dict()
        for inp_name, inp_obj in inputs.items():
            if isinstance(inp_obj, FunctionalBlockWrapper):
                inp_dict[inp_name] = inp_obj.get_input_slot()
            elif isinstance(inp_obj, BlockInputSlot):
                inp_dict[inp_name] = inp_obj
            else:
                raise RuntimeError(f'Input object {inp_name} has wrong type. \
                    FunctionalBlockWrapper and BlockInputSlot are only supported')
        out_dict = dict()
        for out_name, out_obj in outputs.items():
            if isinstance(out_obj, FunctionalBlockWrapper):
                out_dict[out_name] = out_obj.get_output_slot()
            elif isinstance(out_obj, BlockOutputSlot):
                out_dict[out_name] = out_obj
            else:
                raise RuntimeError(f'Output object {out_name} has wrong type. \
                    FunctionalBlockWrapper and BlockOutputSlot are only supported')
        return BasePipeline(self._nodes, self._connections, inp_dict, out_dict)

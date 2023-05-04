from typing import Literal, Mapping, Type, Union
from ...block import BaseBlock, BaseInputBlock, BaseOutputBlock
from ...block.functional import FunctionalBlockWrapper
from ...block.repo import BaseBlockClassRepository, DEFAULT_BLOCK_CLASS_REPOSITORY
from ...block.base import BlockInputSlot, BlockOutputSlot
from ..connection import Connection
from ..base import BasePipeline
from .base import BasePipelineBuilder
from typing import List, Optional, Callable


SlotOrBlockWrapper = Union[BlockInputSlot, FunctionalBlockWrapper]


class FunctionalPipelineBuilder(BasePipelineBuilder):
    """Pipeline builder with functional interface.

    Args:
        block_repo: Block class repository for resolving blocks by their names.
                    Default is zoo scope repository, which means that
                    all blocks defined in :py:mod:`bosk.block.zoo` will be available
                    (without postfix "Block", for example:
                    :py:class:`bosk.block.zoo.data_conversion.ArgmaxBlock`
                    should be accessed as just "Argmax").

    """
    def __init__(self, block_repo: Optional[BaseBlockClassRepository] = None):
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

    def _make_placeholder_fn(self, block: BaseBlock) -> Callable[..., FunctionalBlockWrapper]:
        def placeholder_fn(*pfn_args, **pfn_kwargs) -> FunctionalBlockWrapper:
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
            if len(pfn_args) != 0:
                assert len(pfn_kwargs) == 0, "All arguments should be either named or unnamed"
                assert len(block.slots.inputs), "Please, specify argument names: `block(arg1=value1, ..)`"
                input_block_wrapper = pfn_args[0]
                assert isinstance(input_block_wrapper, FunctionalBlockWrapper), \
                    f'Expected the argument is `FunctionalBlockWrapper` but got {type(input_block_wrapper)}. '\
                    'Maybe wrapping function is passed instead of block placeholder, check '\
                    'that some arguments were passed to the placeholder making function.'
                single_input = next(iter(block.slots.inputs.values()))
                self._connections.append(
                    Connection(
                        src=input_block_wrapper.get_output_slot(),
                        dst=single_input,
                    )
                )
            else:  # no unnamed arguments
                for input_name, input_block_wrapper in pfn_kwargs.items():
                    self._connections.append(
                        Connection(
                            src=input_block_wrapper.get_output_slot(),
                            dst=block.slots.inputs[input_name],
                        )
                    )
            return FunctionalBlockWrapper(block)

        return placeholder_fn

    def wrap(self, block: BaseBlock) -> Callable[..., FunctionalBlockWrapper]:
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

    def _get_block_init(self, block_cls: Type[BaseBlock]) -> Callable[..., Callable[..., FunctionalBlockWrapper]]:
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

    def new(self, block_cls: Type[BaseBlock], *args, **kwargs) -> Callable[..., FunctionalBlockWrapper]:
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

    def build(self, inputs: Mapping[str, SlotOrBlockWrapper] | Literal['auto'] = 'auto',
              outputs: Mapping[str, SlotOrBlockWrapper] | Literal['auto'] = 'auto') -> BasePipeline:
        """Build and get pipeline.

        Args:
            inputs: Dictionary containing the information about pipeline's inputs. See :attr:`BasePipeline.inputs`.
            outputs: Dictionary containing the information about pipeline's outputs. See :attr:`BasePipeline.outputs`

        Returns:
            Pipeline made from wrapped blocks.

        """
        inp_dict = dict()
        if inputs == 'auto':
            for node in self._nodes:
                if isinstance(node, BaseInputBlock):
                    inp_name = node.name
                    if inp_name is None:
                        continue
                    inp_dict[inp_name] = node.get_single_input()
        else:  # inputs is a mapping from name to block wrapper or input slot
            for inp_name, inp_obj in inputs.items():
                if isinstance(inp_obj, FunctionalBlockWrapper):
                    inp_dict[inp_name] = inp_obj.get_input_slot()
                elif isinstance(inp_obj, BlockInputSlot):
                    inp_dict[inp_name] = inp_obj
                else:
                    raise RuntimeError(f'Input object {inp_name} has wrong type. \
                        FunctionalBlockWrapper and BlockInputSlot are only supported')

        out_dict = dict()
        if outputs == 'auto':
            for node in self._nodes:
                if isinstance(node, BaseOutputBlock):
                    out_name = node.name
                    if out_name is None:
                        continue
                    out_dict[out_name] = node.get_single_output()
        else:  # outputs is a mapping from name to block wrapper or output slot
            for out_name, out_obj in outputs.items():
                if isinstance(out_obj, FunctionalBlockWrapper):
                    out_dict[out_name] = out_obj.get_output_slot()
                elif isinstance(out_obj, BlockOutputSlot):
                    out_dict[out_name] = out_obj
                else:
                    raise RuntimeError(f'Output object {out_name} has wrong type. \
                        FunctionalBlockWrapper and BlockOutputSlot are only supported')
        return BasePipeline(nodes=self._nodes, connections=self._connections, inputs=inp_dict, outputs=out_dict)

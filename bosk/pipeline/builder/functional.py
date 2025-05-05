from dataclasses import dataclass
from ...block.placeholder import PlaceholderMixin
from ...block import BaseBlock, BaseInputBlock, BaseOutputBlock
from ...block.functional import FunctionalBlockWrapper
from ...block.repo import BaseBlockClassRepository, DEFAULT_BLOCK_CLASS_REPOSITORY
from ...block.base import BlockInputSlot, BlockOutputSlot
from ...block.zoo.routing.shared import SharedProducer, SharedConsumer
from ..connection import Connection
from ..base import BasePipeline
from .base import BasePipelineBuilder
from ...exceptions import BlockReuseError
import inspect
from typing import Dict, Literal, Mapping, Set, Tuple, Type, Union
from typing import List, Optional, Callable
from collections import defaultdict


SlotOrBlockWrapper = Union[BlockInputSlot, FunctionalBlockWrapper]


@dataclass
class PlaceholderFunctionState:
    block: BaseBlock


class FunctionalPipelineBuilder(BasePipelineBuilder):
    """Pipeline builder with functional interface.

    Args:
        block_repo: Block class repository for resolving blocks by their names.
                    Default is zoo scope repository, which means that
                    all blocks defined in :py:mod:`bosk.block.zoo` will be available
                    (without postfix "Block", for example:
                    :py:class:`bosk.block.zoo.data_conversion.ArgmaxBlock`
                    should be accessed as just "Argmax").
        allow_block_reuse: If True, a block can be reused (which is implemented
                           as replacing it in pipeline with SharedProducer or
                           SharedConsumer), otherwise the exception `BlockReuseError`
                           will be thrown if a block is encountered more than once.

    """

    def __init__(self, block_repo: Optional[BaseBlockClassRepository] = None, allow_block_reuse: bool = True):
        self._nodes: List[BaseBlock] = []
        self._connections: List[Connection] = []
        if block_repo is None:
            block_repo = DEFAULT_BLOCK_CLASS_REPOSITORY
        self._block_repo: BaseBlockClassRepository = block_repo
        self._called_blocks: Set[BaseBlock] = set()
        self._block_wrappers: Dict[BaseBlock,
                                   List[FunctionalBlockWrapper]] = defaultdict(list)
        self._wrapped_states: Dict[BaseBlock,
                                   PlaceholderFunctionState] = dict()
        self.allow_block_reuse = allow_block_reuse

    def __getattr__(self, name: str) -> Callable:
        block_cls = self._block_repo.get(name)
        return self._get_block_init(block_cls)

    def _register_block(self, block: BaseBlock):
        """Register block in the builder.

        Args:
            block: Block to register.

        """
        self._nodes.append(block)

    def _replace_block(self, replaced: BaseBlock, new: BaseBlock):
        """Replace an existing block in the pipeline with a new one.

        This method is used when a block needs to be updated or modified after it has already been added to the pipeline.

        It's particularly useful in scenarios involving block reuse,
        where a shared producer or consumer replaces a previously defined block instance.

        Args:
            replaced: The BaseBlock that needs to be replaced.
            new: The new BaseBlock that will replace the old one.

        """
        # replace the block in nodes
        replaced_index = self._nodes.index(replaced)
        self._nodes[replaced_index] = new
        # recreate connections
        new_connections = []
        for conn in self._connections:
            if id(conn.src.parent_block) == id(replaced):
                new_connections.append(Connection(
                    src=new.slots.outputs[conn.src.meta.name],
                    dst=conn.dst,
                ))
            elif id(conn.dst.parent_block) == id(replaced):
                new_connections.append(Connection(
                    src=conn.src,
                    dst=new.slots.inputs[conn.dst.meta.name],
                ))
            else:
                new_connections.append(conn)
        self._connections = new_connections
        # replace wrappers
        block_wrappers = self._block_wrappers[replaced]
        for wrapper in block_wrappers:
            wrapper.set_block(new)
        self._block_wrappers[new] = block_wrappers
        del self._block_wrappers[replaced]

    def _handle_block_reuse(self, state: PlaceholderFunctionState) -> Tuple[BaseBlock, Optional[SharedProducer]]:
        """Handles the reuse of a block in the pipeline.

        If `allow_block_reuse` is True, replaces the existing block with a SharedProducer or SharedConsumer.
        Otherwise, raises a BlockReuseError if the block has already been encountered.

        Args:
            state: The current placeholder function state.

        Returns:
            A tuple containing the replaced block (either the original or a shared producer/consumer),
            and an optional SharedProducer instance.

            If `allow_block_reuse` is False, the first element of the tuple will be the original block,
            and the second element will be None.
            If `allow_block_reuse` is True, the first element will be either the original block
            (if it is called once) or a new SharedConsumer instance,
            and the second element will be the SharedProducer instance itself.

        """
        block = state.block
        producer_block: Optional[SharedProducer] = None
        if block in self._called_blocks or isinstance(block, SharedProducer):
            if not self.allow_block_reuse:
                raise BlockReuseError(block)
            if not isinstance(block, SharedProducer):
                producer_block = SharedProducer(
                    block, output_block_name=f'__block_{id(block)}')
                self._replace_block(block, producer_block)
                state.block = producer_block
            assert isinstance(state.block, SharedProducer)
            producer_block = state.block
            # current block becomes the state consumer
            block = SharedConsumer(
                producer_block.meta,
                input_block_name=producer_block.output_block_name,
                default_output_name=producer_block.get_default_output().meta.name,
            )
            self._nodes.append(block)
            self._connections.append(
                Connection(
                    src=producer_block.slots.outputs[producer_block.output_block_name],
                    dst=block.slots.inputs[producer_block.output_block_name],
                )
            )
        return block, producer_block

    def _register_block_wrapper(self, block: BaseBlock, block_wrapper: FunctionalBlockWrapper):
        """Register block wrapper.

        Registers the result of a placeholder function call, allowing to track all
        used block wrappers.

        Args:
            block: The called block.
            block_wrapper: The created block wrapper.

        """
        self._called_blocks.add(block)
        self._block_wrappers[block].append(block_wrapper)

    def _make_placeholder_fn(self, block: BaseBlock,
                             state: Optional[PlaceholderFunctionState] = None) -> Callable[..., FunctionalBlockWrapper]:
        if state is None:
            state = PlaceholderFunctionState(block)

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
            block, producer_block = self._handle_block_reuse(state)

            if len(pfn_args) != 0:
                assert len(
                    pfn_kwargs) == 0, "All arguments should be either named or unnamed"
                assert len(
                    block.slots.inputs), "Please, specify argument names: `block(arg1=value1, ..)`"
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
            # on success update call history
            block_wrapper = FunctionalBlockWrapper(block)
            self._register_block_wrapper(block, block_wrapper)
            return block_wrapper

        # make appropriate docstring and function signature
        placeholder_fn.__doc__ = f'Execute the block {block} and get the result wrapper.' \
            '\n\n' + ('=' * (25 + len(repr(block)))) + '\n' \
            f'The block {block} documentation:\n\n' + \
            str(inspect.getdoc(block.__class__))
        setattr(
            placeholder_fn,
            '__signature__',
            inspect.Signature([
                inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY)
                for name in block.slots.inputs.keys()
            ])
        )
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
        if block not in self._wrapped_states:
            self._register_block(block)
            self._wrapped_states[block] = PlaceholderFunctionState(block)
        state = self._wrapped_states[block]
        return self._make_placeholder_fn(block, state=state)

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

    def build(self, inputs: Union[Mapping[str, SlotOrBlockWrapper], Literal['auto']] = 'auto',
              outputs: Union[Mapping[str, SlotOrBlockWrapper], Literal['auto']] = 'auto') -> BasePipeline:
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

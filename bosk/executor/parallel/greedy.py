from collections import defaultdict
from typing import Callable, Dict, Iterable, Mapping, Set, TypeVar, Union, Sequence, Optional, List

from ...data import BaseData, Data
from ..base import BaseBlockExecutor, BaseExecutor, BaseSlotHandler, Stage
from ...pipeline import BasePipeline
from ...block.base import BaseBlock, BlockOutputData, BlockInputSlot, BlockOutputSlot
from ..utility import get_connection_map
from joblib import Parallel as JoblibParallel, delayed as joblib_delayed
from multiprocessing.pool import ThreadPool as MultiprocessingThreadPool
from abc import ABC, abstractmethod


ResultT = TypeVar('ResultT')
"""Execution result generic typevar.

Required to constrain `ParallelEngine.Instance.staramap` result.
"""


class ParallelEngine(ABC):
    """Parallel execution engine interface.
    """

    class Instance(ABC):
        """Execution engine instance interface.
        """
        @abstractmethod
        def starmap(self, func: Callable[..., ResultT], iterable: Iterable) -> List[ResultT]:
            ...

    @abstractmethod
    def __enter__(self) -> 'ParallelEngine.Instance':
        ...

    @abstractmethod
    def __exit__(self, _type, _value, _traceback):
        ...


class JoblibParallelEngine(ParallelEngine):
    """Joblib-based Parallel Engine.
    """
    class JoblibInstance(ParallelEngine.Instance):
        def __init__(self, parallel: JoblibParallel):
            self.parallel = parallel

        def starmap(self, func, iterable):
            return self.parallel(
                joblib_delayed(func)(*args)
                for args in iterable
            )

    def __init__(self, n_threads: int = -1,
                 backend: Optional[str] = None,
                 prefer: Optional[str] = 'threads'):
        self.n_threads = n_threads
        self.backend = backend
        self.prefer = prefer

    def __enter__(self) -> 'JoblibParallelEngine.JoblibInstance':
        self.pool_instance = JoblibParallel(
            self.n_threads,
            backend=self.backend,
            prefer=self.prefer,
        )
        return self.JoblibInstance(self.pool_instance.__enter__())

    def __exit__(self, _type, _value, _traceback):
        self.pool_instance.__exit__(_type, _value, _traceback)


class MultiprocessingParallelEngine(ParallelEngine):
    """Multiprocessing-based thread pool execution engine.
    """
    class MPInstance(ParallelEngine.Instance):
        def __init__(self, pool: MultiprocessingThreadPool):
            self.pool = pool

        def starmap(self, func, iterable):
            return self.pool.starmap(func, iterable)

    def __init__(self, n_threads: Optional[int] = None):
        self.n_threads = n_threads

    def __enter__(self) -> 'MultiprocessingParallelEngine.MPInstance':
        self.pool_instance = MultiprocessingThreadPool(self.n_threads)
        return self.MPInstance(self.pool_instance.__enter__())

    def __exit__(self, _type, _value, _traceback):
        self.pool_instance.__exit__(_type, _value, _traceback)


class GreedyParallelExecutor(BaseExecutor):
    """The recursive executor implementation.

    Considers only input-output slots information to match slots.

    Attributes:
        _conn_map: Pipeline connections, represented as a hash map, the keys are blocks' input slots,
            the values are output ones. Each input slot corresponds no more than one
            output slot, so this representation is correct.

    Args:
        pipeline: Sets :attr:`.BaseExecutor.__pipeline`.
        stage: Sets :attr:`.BaseExecutor.__stage`,
        inputs: Sets :attr:`.BaseExecutor.__inputs`.
        outputs: Sets :attr:`.BaseExecutor.__outputs`.
        slot_handler: Sets :attr:`.BaseExecutor.__slot_handler` with `_prepare_slot_handler` method.
        block_executor: Sets :attr:`.BaseExecutor.__block_executor` with `_prepare_block_executor` method.
    """

    _conn_map: Mapping[BlockInputSlot, BlockOutputSlot]

    def __init__(self, pipeline: BasePipeline,
                 stage: Stage,
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None,
                 slot_handler: Optional[BaseSlotHandler] = None,
                 block_executor: Optional[BaseBlockExecutor] = None,
                 parallel_engine: ParallelEngine = MultiprocessingParallelEngine()) -> None:
        super().__init__(pipeline, stage, inputs, outputs, slot_handler, block_executor)
        self.parallel_engine = parallel_engine
        self._conn_map = get_connection_map(self)
        self._edges = self._prepare_out_to_in_edges()
        self._inputs_by_block = self._prepare_inputs_by_block()

    def _prepare_out_to_in_edges(self) -> Dict[BlockOutputSlot, List[BlockInputSlot]]:
        """Prepare the mapping from output slots to list of input slots.

        Returns:
            Dictionary with output slots as keys,
            lists of the corresponding input slots as values.
        """
        result = defaultdict(list)
        for conn in self.pipeline.connections:
            result[conn.src].append(conn.dst)
        return result

    def _get_blocks(self, output_slots: Set[BlockOutputSlot]) -> Set[BaseBlock]:
        """Get all blocks that should be executed.

        Args:
            output_slots: Set of output slots.

        Returns:
            Set of the pipeline blocks.

        """
        dst_to_src = dict()
        for conn in self.pipeline.connections:
            dst_to_src[conn.dst] = conn.src
        # reverse DFS over block output slots
        stack = [slot for slot in output_slots]
        result_blocks = set()
        while stack:
            out_slot = stack.pop()
            block = out_slot.parent_block
            if block in result_blocks:
                continue
            result_blocks.add(block)
            for input_slot in block.slots.inputs.values():
                if self._is_slot_required(input_slot) and input_slot in dst_to_src:
                    stack.append(dst_to_src[input_slot])
        return result_blocks

    def _prepare_inputs_by_block(self) -> Dict[BaseBlock, Set[BlockInputSlot]]:
        """Prepare the mapping from blocks to their inputs.

        Returns:
            Dictionary with blocks as keys,
            sets of the corresponding input slots as values.

        """
        result = defaultdict(set)
        for conn in self.pipeline.connections:
            result[conn.dst.parent_block].add(conn.dst)
        return result

    def _prepare_inputs(self, block,
                        input_slot_values: Mapping[BlockInputSlot, Data]) -> Mapping[BlockInputSlot, Data]:
        """Prepare the mapping of inputs needed for the block.

        Args:
            block: The block for which input values are needed.
            input_slot_values: Mapping from input slots to the corresponding data.

        Returns:
            Mapping from input slots to the corresponding data for the given block.

        """
        block_input_slots = block.slots.inputs.values()
        return {
            input_slot: input_slot_values[input_slot]
            for input_slot in block_input_slots
            if input_slot in input_slot_values
        }

    def _compute_all_plain(self, blocks: Sequence[BaseBlock],
                           computed_values: Mapping[BlockInputSlot, Data]) -> Mapping[BlockOutputSlot, Data]:
        """Filter plain blocks and compute them.

        It is assumed that plain block execution is computationally effortless.

        Args:
            blocks: Blocks that potentially can be computed (not necessarily plain).

        Returns:
            Mapping from `BlockOutputSlot` to `Data`.

        """
        outputs: Dict[BlockOutputSlot, Data] = dict()
        for block in blocks:
            if not block.meta.execution_props.plain:
                continue
            block_inputs = self._prepare_inputs(block, computed_values)
            block_outputs = self._execute_block(block, block_inputs)
            outputs.update(block_outputs)
        return outputs

    def _compute_all_parallel(self, blocks: Sequence[BaseBlock],
                              computed_values: Mapping[BlockInputSlot, Data],
                              parallel: ParallelEngine.Instance) -> Mapping[BlockOutputSlot, Data]:
        """Filter blocks that can be computed in parallel and compute them.

        Args:
            blocks: All blocks that potentially can be computed.

        Returns:
            Mapping from `BlockOutputSlot` to `Data`.

        """
        outputs: Dict[BlockOutputSlot, Data] = dict()
        # TODO: consider another impl: call block executor method like `execute_threadsafe_blocks`
        #   in this case execution behaviour will be implemented in block executor
        parallel_results = parallel.starmap(
            self._execute_block, (
                (block, self._prepare_inputs(block, computed_values))
                for block in blocks
                if block.meta.execution_props.threadsafe and not block.meta.execution_props.plain
            )
        )
        for block_output in parallel_results:
            outputs.update(block_output)
        return outputs

    def _compute_all_non_threadsafe(self, blocks: Sequence[BaseBlock],
                                    computed_values: Mapping[BlockInputSlot, Data]) -> Mapping[BlockOutputSlot, Data]:
        """Filter blocks that are not plain and cannot be computed in parallel, and compute them.

        Args:
            blocks: All blocks that potentially can be computed.

        Returns:
            Mapping from `BlockOutputSlot` to `Data`.

        """
        outputs: Dict[BlockOutputSlot, Data] = dict()
        for block in blocks:
            if block.meta.execution_props.threadsafe or \
                    block.meta.execution_props.plain:
                continue
            block_inputs = self._prepare_inputs(block, computed_values)
            block_outputs = self._execute_block(block, block_inputs)
            outputs.update(block_outputs)
        return outputs

    def _clean_unnecessary_data(self, computed_values: Dict[BlockInputSlot, Data],
                                remaining_blocks: Set[BaseBlock]):
        """Remove the intermediate data (execution results) that will not be required in the future.

        Args:
            computed_values: Dictionary of already computed values.
            remaining_blocks: Set of blocks that should be computed in the next steps.

        Returns:

        """
        to_delete = []
        for in_slot in computed_values.keys():
            if in_slot.parent_block not in remaining_blocks:
                to_delete.append(in_slot)
        for in_slot in to_delete:
            del computed_values[in_slot]

    def _find_ready_blocks(self, computed_values: Dict[BlockInputSlot, Data],
                           remaining_blocks: Set[BaseBlock]) -> List[BaseBlock]:
        """Find the blocks for which required inputs are already computed.

        Args:
            computed_values: Mapping from input slots to the corresponding computed data.
            remaining_blocks: Set of blocks which haven't been computed yet.

        Returns:
            List of blocks which are ready to be computed.

        """
        result = []
        # the following would work only if optional inputs are be labeled (as optional)
        # ... and `self._is_slot_required` for such slots is False
        # for block in remaining_blocks:
        #     for slot in block.slots.inputs.values():
        #         if slot not in computed_values and self._is_slot_required(slot):
        #             break
        #     else:
        #         result.append(block)
        for block in remaining_blocks:
            for slot in self._inputs_by_block[block]:
                if slot not in computed_values and self._is_slot_required(slot):
                    break
            else:
                result.append(block)
        return result

    def __append_outputs(self, output_values: Dict[BlockOutputSlot, Data],
                         computed_values: Dict[BlockInputSlot, Data],
                         output_slots: Set[BlockOutputSlot],
                         new_outputs: BlockOutputData):
        """Append newly computed outputs.

        Args:
            output_values: Final output values (will be modified).
            computed_values: Computed values required for following blocks computation (will be modified).
            output_slots: Set of output slots.
            new_outputs: Newly computed outputs.

        """
        for out_slot, out_data in new_outputs.items():
            if out_slot in output_slots:
                # append data to the final output values
                output_values[out_slot] = out_data
            # append data to computed values
            for in_slot in self._edges[out_slot]:
                computed_values[in_slot] = out_data

    def __execute_with_parallel(self, input_values: Mapping[str, Data],
                             parallel: ParallelEngine.Instance) -> Dict[BlockOutputSlot, BaseData]:
        """Pipeline execution with given parallel engine instance.

        Args:
            input_values: Input values data mapping.
            parallel: Parallel engine instance.

        Returns:
            Dictionary with output slots as keys and computed data as values.

        """
        initial_input_slot_values = self._map_input_names_to_slots(input_values)
        assert self.outputs is not None
        output_slots = {
            out_slot
            for out_name, out_slot in self.pipeline.outputs.items()
            if out_name in self.outputs
        }
        output_values: Dict[BlockOutputSlot, Data] = dict()
        remaining_blocks: Set[BaseBlock] = self._get_blocks(output_slots)
        computed_blocks: Set[BaseBlock] = set()

        # Iteratively:
        # 1. Determine which blocks can be computed given input slot values.
        # 2. Compute plain (straightforward) blocks.
        #    Go to 1 if some blocks were computed.
        # 3. Compute non-threadsafe non-plain blocks.
        #    Go to 1 if some blocks were computed.
        # 4. Compute in parallel threadsafe blocks.

        computed_values: Dict[BlockInputSlot, Data] = initial_input_slot_values
        recently_computed_outputs: Optional[BlockOutputData] = None
        while True:
            if recently_computed_outputs is not None:
                self.__append_outputs(
                    output_values,
                    computed_values,
                    output_slots,
                    recently_computed_outputs
                )
                computed_blocks.update(
                    out.parent_block
                    for out in recently_computed_outputs.keys()
                )
                remaining_blocks.difference_update(computed_blocks)
                recently_computed_outputs = None

            if len(output_values) == len(output_slots):  # all output values have been computed
                break

            # Determine which blocks can be computed.
            ready_blocks = self._find_ready_blocks(computed_values, remaining_blocks)
            # Remove the already computed values which won't be used in the future
            self._clean_unnecessary_data(computed_values, remaining_blocks)
            if len(ready_blocks) == 0:
                # no blocks to compute found => output cannot be calculated
                raise RuntimeError(f"No way to compute {output_slots}")
            # Compute plain blocks
            plain_outputs = self._compute_all_plain(ready_blocks, computed_values)
            if len(plain_outputs) > 0:
                recently_computed_outputs = plain_outputs
                continue
            # Compute non-threadsafe non-plain blocks
            non_threadsafe_outputs = self._compute_all_non_threadsafe(ready_blocks, computed_values)
            if len(non_threadsafe_outputs) > 0:
                recently_computed_outputs = non_threadsafe_outputs
                continue
            # Compute blocks in parallel
            parallel_outputs = self._compute_all_parallel(ready_blocks, computed_values, parallel)
            recently_computed_outputs = parallel_outputs
        return output_values

    def execute(self, input_values: Mapping[str, Data]) -> Dict[str, BaseData]:
        self._check_input_values(input_values)
        with self.parallel_engine as parallel:
            output_values = self.__execute_with_parallel(input_values, parallel)
        # convert output values to string-indexed dict
        result = dict()
        assert self.outputs is not None
        for output_name, output_slot in self.pipeline.outputs.items():
            if output_name in self.outputs:
                result[output_name] = output_values[output_slot]
        return result

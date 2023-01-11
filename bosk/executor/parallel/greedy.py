from collections import defaultdict
from typing import Dict, Mapping, Set, Union, Sequence, Optional, List

from ...data import Data
from ..base import BaseExecutor
from ...pipeline import BasePipeline
from ..descriptor import HandlingDescriptor
from ...block.slot import BlockInputSlot, BlockOutputSlot
from ...block.base import BaseBlock, BlockOutputData
from ..utility import get_connection_map
from joblib import Parallel, delayed
from multiprocessing.pool import ThreadPool
from abc import ABC, abstractmethod


class ParallelEngine(ABC):
    class Instance:
        def __init__(self):
            pass

    @abstractmethod
    def __enter__(self) -> 'ParallelEngine.Instance':
        ...

    @abstractmethod
    def __exit__(self, _type, _value, _traceback):
        ...


class JoblibParallelEngine(ParallelEngine):
    class JoblibInstance(ParallelEngine.Instance):
        def __init__(self, parallel: Parallel):
            self.parallel = parallel

        def starmap(self, func, iterable):
            return self.parallel(
                delayed(func)(*args)
                for args in iterable
            )

    def __init__(self, n_threads: int = -1,
                 backend: Optional[str] = None,
                 prefer: Optional[str] = 'threads'):
        self.n_threads = n_threads
        self.backend = backend
        self.prefer = prefer

    def __enter__(self) -> 'JoblibParallelEngine.JoblibInstance':
        self.pool_instance = Parallel(
            self.n_threads,
            backend=self.backend,
            prefer=self.prefer,
        )
        return self.JoblibInstance(self.pool_instance.__enter__())

    def __exit__(self, _type, _value, _traceback):
        self.pool_instance.__exit__(_type, _value, _traceback)


class ThreadingParallelEngine(ParallelEngine):
    class TPEInstance(ParallelEngine.Instance):
        def __init__(self, pool: ThreadPool):
            self.pool = pool

        def starmap(self, func, iterable):
            return self.pool.starmap(func, iterable)

    def __init__(self, n_threads: Optional[int] = None):
        self.n_threads = n_threads

    def __enter__(self) -> 'ThreadingParallelEngine.TPEInstance':
        self.pool_instance = ThreadPool(self.n_threads)
        return self.TPEInstance(self.pool_instance.__enter__())

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
        stage_descriptor: Sets :attr:`.BaseExecutor.__stage`,
            :attr:`.BaseExecutor.__slots_handler` and :attr:`.BaseExecutor.__blocks_handler`.
        inputs: Sets :attr:`.BaseExecutor.__inputs`.
        outputs: Sets :attr:`.BaseExecutor.__outputs`.
    """

    _conn_map: Mapping[BlockInputSlot, BlockOutputSlot]

    def __init__(self, pipeline: BasePipeline,
                 handl_desc: HandlingDescriptor,
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None,
                 parallel_engine: ParallelEngine = ThreadingParallelEngine()) -> None:
        super().__init__(pipeline, handl_desc, inputs, outputs)
        self.parallel_engine = parallel_engine
        self._conn_map = get_connection_map(self)
        self._edges = self._prepare_out_to_in_edges()
        self._inputs_by_block = self._prepare_inputs_by_block()

    def _prepare_out_to_in_edges(self) -> Dict[BlockOutputSlot, List[BlockInputSlot]]:
        result = defaultdict(list)
        for conn in self.pipeline.connections:
            result[conn.src].append(conn.dst)
        return result

    def _get_blocks(self) -> Set[BaseBlock]:
        return set(
            b
            for conn in self.pipeline.connections
            for b in (conn.src.parent_block, conn.dst.parent_block)
        )

    def _prepare_inputs_by_block(self) -> Dict[BaseBlock, Set[BlockInputSlot]]:
        result = defaultdict(set)
        for conn in self.pipeline.connections:
            result[conn.dst.parent_block].add(conn.dst)
        return result

    def _prepare_inputs(self, block,
                        input_slot_values: Mapping[BlockInputSlot, Data]) -> Mapping[BlockInputSlot, Data]:
        block_input_slots = block.slots.inputs.values()
        return {
            input_slot: input_slot_values[input_slot]
            for input_slot in block_input_slots
            if input_slot in input_slot_values
        }

    def _compute_all_plain(self, blocks: Sequence[BaseBlock],
                           computed_values: Mapping[BlockInputSlot, Data]) -> Mapping[BlockOutputSlot, Data]:
        """Filter plain blocks and compute them.

        Args:
            blocks: Blocks that potentially can be computed (not necessarily plain).

        Returns:
            Mapping from `BlockOutputSlot` to `Data`.

        """
        outputs = dict()
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
        outputs = dict()
        # TODO: replace with calling block executor method like `execute_threadsafe_blocks`
        #   in this case execution behaviour will be implemented in block executor
        # parallel_results = parallel(
        #     delayed(self._execute_block)(block, self._prepare_inputs(block, computed_values))
        #     for block in blocks
        #     if block.meta.execution_props.threadsafe and not block.meta.execution_props.plain
        # )
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
        outputs = dict()
        for block in blocks:
            if block.meta.execution_props.threadsafe or \
                block.meta.execution_props.plain:
                continue
            block_inputs = self._prepare_inputs(block, computed_values)
            block_outputs = self._execute_block(block, block_inputs)
            outputs.update(block_outputs)
        return outputs

    def _clean_unnecessary_data(self, computed_values: Dict[BlockInputSlot, Data],
                                remaining_blocks: Set[BaseBlock]) -> Sequence[BaseBlock]:
        to_delete = []
        for in_slot in computed_values.keys():
            if in_slot.parent_block not in remaining_blocks:
                to_delete.append(in_slot)
        for in_slot in to_delete:
            del computed_values[in_slot]

    def _find_ready_blocks(self, computed_values: Dict[BlockInputSlot, Data],
                          remaining_blocks: Set[BaseBlock]) -> Sequence[BaseBlock]:
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
        """
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

    def __call_with_parallel(self, input_values: Mapping[str, Data],
                             parallel: ParallelEngine.Instance) -> Dict[BlockOutputSlot, Data]:
        initial_input_slot_values = self._map_input_names_to_slots(input_values)
        output_slots = set(self.pipeline.outputs.values())
        output_values: Dict[BlockOutputSlot, Data] = dict()
        remaining_blocks: Set[BaseBlock] = self._get_blocks()
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

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        self._check_input_values(input_values)
        with self.parallel_engine as parallel:
            output_values = self.__call_with_parallel(input_values, parallel)
        # convert output values to string-indexed dict
        result = dict()
        for output_name, output_slot in self.pipeline.outputs.items():
            result[output_name] = output_values[output_slot]
        return result

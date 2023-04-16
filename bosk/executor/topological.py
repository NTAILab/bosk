"""Topological executor with the graph painter.

This file contains the optimizing executor, which also can draw the computational graph.

"""

from typing import Deque, Dict, List, Mapping, Sequence, Set, Optional
from collections import defaultdict, deque

from ..data import Data
from .base import BaseBlockExecutor, BaseExecutor, BaseSlotHandler, Stage
from ..pipeline import BasePipeline
from ..block.slot import BaseSlot, BlockInputSlot, BlockOutputSlot
from ..block import BaseBlock
from .utility import get_connection_map

import warnings


class TopologicalExecutor(BaseExecutor):
    """Topological executor with the graph painter.
    The optimization algoritm computes only blocks which are connected with the inputs and needed for
    the outputs calculation.

    Attributes:
        _conn_dict: Pipeline connections, represented as a hash map, the keys are blocks' input slots,
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

    _conn_dict: Mapping[BlockInputSlot, BlockOutputSlot]

    def __init__(self, pipeline: BasePipeline,
                 stage: Stage,
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None,
                 slot_handler: Optional[BaseSlotHandler] = None,
                 block_executor: Optional[BaseBlockExecutor] = None) -> None:
        super().__init__(pipeline, stage, inputs, outputs, slot_handler, block_executor)
        conn_dict = get_connection_map(self)
        for inp in self.pipeline.inputs.values():
            conn_dict[inp] = inp
        self._conn_dict = conn_dict

    def _dfs(self, aj_list: Mapping[BaseBlock, Set[BaseBlock]], begin_nodes: Sequence[BaseBlock]) -> Set[BaseBlock]:
        """Method that performs the deep first search algorithm in the computational graph.
        The search begins from the nodes `begin_nodes`. The algorithm is written using iterative scheme.

        Args:
            aj_list: The graph adjacency list.
            begin_nodes: The graph nodes, which will be used as the search start.

        Returns:
            Set of the graph nodes, which were included in the traversal. Actually, the order is not saved.
            This method is only for internal use, so this behaviour is chosen for the optimization.
        """
        visited: Set[BaseBlock] = set(begin_nodes)
        stack: Deque[BaseBlock] = deque(begin_nodes)
        while stack:
            node = stack.pop()
            for neig_node in aj_list[node]:
                if neig_node not in visited:
                    visited.add(neig_node)
                    stack.append(neig_node)
        return visited

    def _topological_sort(self, aj_list: Mapping[BaseBlock, Set[BaseBlock]], begin_nodes: Sequence[BaseBlock]) -> List[BaseBlock]:
        """Method that performs the topological sort of the computational graph.
        The algorithm begins its work from the nodes `begin_nodes`. The algorithm is written using recursive scheme.

        Args:
            aj_list: The graph adjacency list.
            begin_nodes: The graph nodes, which will be used as the algorithm start.

        Returns:
            List of the graph nodes in topological order.

        """
        visited: Set[BaseBlock] = set()
        outer_stack: Deque[BaseBlock] = deque(begin_nodes)
        inner_stack: List[BaseBlock] = list()
        order: List[BaseBlock] = list()

        while outer_stack:
            node = outer_stack.pop()
            if node not in visited:
                visited.add(node)
                outer_stack.extend(aj_list[node])
                while inner_stack and node not in aj_list[inner_stack[-1]]:
                    order.append(inner_stack.pop())
                inner_stack.append(node)

        inner_stack.extend(reversed(order))
        return inner_stack

    def _get_backward_aj_list(self, feasible_set: None | Set[BaseBlock] = None) -> Mapping[BaseBlock, Set[BaseBlock]]:
        """The internal helper method for making backwards adjacency list
        (block to block, from the end to the start) of the pipeline.

        Args:
            feasible_set: The set of the blocks, which will be used in the adjacency list.
                Other blocks will not be included. If `None`, all blocks of the pipeline will be used.

        Returns:
            The backwards adjacency list containing blocks from the ``feasible set``.
        """
        backward_aj_list: Mapping[BaseBlock, Set[BaseBlock]] = defaultdict(set)
        for inp_slot, out_slot in self._conn_dict.items():
            if feasible_set is None or \
               inp_slot.parent_block in feasible_set and out_slot.parent_block in feasible_set:
                backward_aj_list[inp_slot.parent_block].add(out_slot.parent_block)
        return backward_aj_list

    def _get_forward_aj_list(self, feasible_set: None | Set[BaseBlock] = None) -> Mapping[BaseBlock, Set[BaseBlock]]:
        """The internal helper method for making adjacency list (block to block) of the pipeline.

        Args:
            feasible_set: The set of the blocks, which will be used in the adjacency list.
                Other blocks will not be included. If `None`, all blocks of the pipeline will be used.

        Returns:
            The adjacency list containing blocks from the ``feasible set``.
        """
        forward_aj_list: Mapping[BaseBlock, Set[BaseBlock]] = defaultdict(set)
        for inp_slot, out_slot in self._conn_dict.items():
            if feasible_set is None or \
               inp_slot.parent_block in feasible_set and out_slot.parent_block in feasible_set:
                forward_aj_list[out_slot.parent_block].add(inp_slot.parent_block)
        return forward_aj_list

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        """The main method for the processing of the computational graph.

        Args:
            input_values: The dictionary, containing the pipeline's inputs names as keys
                and coresponding to them :data:`Data` as values.

        Returns:
            The dictionary, containing the pipeline's outputs names as keys
            and coresponding to them :data:`Data` as values.

        Raises:
            AssertionError: If there are some incompatibility between pipeline's inputs and user's ones.

        """
        self._check_input_values(input_values)

        if self.outputs is not None:
            out_slots_to_process = [self.pipeline.outputs[name] for name in self.outputs]
        else:
            out_slots_to_process = self.pipeline.outputs.values()
        output_blocks = [slot.parent_block for slot in out_slots_to_process]
        backward_pass = self._dfs(self._get_backward_aj_list(), output_blocks)

        slots_values: Dict[BaseSlot, Data] = dict()
        input_blocks_set = set()
        for input_name, input_data in input_values.items():
            input_slot = self.pipeline.inputs.get(input_name, None)
            if input_slot is None:
                continue
            slots_values[input_slot] = input_data
            if input_slot.parent_block not in backward_pass:
                warnings.warn(
                    f'Input slot {input_name!r} of {input_slot.parent_block!r} '
                    "is disconnected from the outputs, it won't be calculated")
            else:
                input_blocks_set.add(input_slot.parent_block)
        topological_order = self._topological_sort(
            self._get_forward_aj_list(backward_pass), input_blocks_set)

        for node in topological_order:
            node_input_data = dict()
            for name, inp_slot in node.slots.inputs.items():
                if inp_slot not in self._conn_dict:
                    # input slot was not used
                    continue
                corresponding_output = self._conn_dict[inp_slot]
                # how about skipping, for example, in concatBlock?
                if corresponding_output not in slots_values:
                    raise RuntimeError(
                        f'The output {corresponding_output} has not been computed, '
                        f'it is needed for the block {node!r}'
                    )
                inp_data = slots_values[corresponding_output]
                node_input_data[inp_slot] = inp_data
            outputs = self._execute_block(node, node_input_data)
            slots_values.update(outputs)

        result: Mapping[str, Data] = dict()
        for output_name, output_slot in self.pipeline.outputs.items():
            if self.outputs is None or output_name in self.outputs:
                slot_data = slots_values.get(output_slot)
                result[output_name] = slot_data
        return result

"""Topological executor with the graph painter.

This file contains the optimizing executor, which also can draw the computational graph.

"""

from typing import Deque, Dict, List, Mapping, Sequence, Set
from collections.abc import Iterable
from collections import defaultdict, deque

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from .painter import PainterMixin
from ..slot import BlockInputSlot, BlockOutputSlot, BaseSlot
from ..block import BaseBlock
from ..pipeline import BasePipeline

import graphviz as gv


class TopologicalExecutor(BaseExecutor, PainterMixin):
    """Topological executor with the graph painter.
    The optimization algoritm computes only blocks which are connected with the inputs and needed for 
    the outputs calculation. You can see which blocks would be executed if you draw the computational graph
    with the 'draw' method.
    
    Attributes:
        conn_dict (Mapping[BlockInputSlot, BlockOutputSlot]): Pipeline connections, represented as a hash map,
            the keys are blocks' input slots, the values are output ones. Each input slot corresponds no more than one 
            output slot, so this representation is correct.
        slot_to_block_map (Mapping[BaseSlot, BaseBlock]): Dictionary that allows to determine to which block
            some slot does belong. Will be removed in the future versions, when the BaseSlot will contain the link to his parent block.
        levels_sep (float): The painter's parameter, which determines the distance between the computational graph's levels. See http://graphviz.org/docs/attrs/ranksep/.
        dpi (int): The dpi of the output computational graph images, formated in raster graphics (.png, .jpeg, etc.).
        rankdir (str): The direction of the computational graph edges. See https://graphviz.org/docs/attrs/rankdir/.
    
    Args:
            pipeline: The pipeline of the computational graph.
            stage: The stage which will be performed by the executor.
            inputs: The dictionary, containing input names as keys and
                block input slots as values. Sets start points of the graph computation.
            outputs: The dictionary, containing output names as keys and
                block output slots as values. Sets end points of the graph computation.
            painter_levels_sep: Sets :attr:`levels_sep`.
            figure_dpi: Sets :attr:`dpi`.
            figure_rankdir: Sets :attr:`rankdir`.
    """

    def __check_inputs_concordance(self, input_values: Mapping[str, Data]) -> None:
        """The function that checks if the input values, provided to the :meth:`__call__` method, agree with the pipeline's inputs.

        Args:
            input_values: Values, which were provided in the :meth:`__call__` method.
        
        Raises:
            AssertionError: If there are some incompatibility between pipeline's inputs and user's ones.
        """
        passed = 0
        for inp_name, inp_slot in self.inputs.items():
            assert (inp_name in input_values), f"Unable to find input slot {inp_name} (id {hash(inp_slot)}) in input data"
            passed += 1
        assert (passed == len(input_values)), "Input values are incompatible with pipeline input slots"

    def _get_slot_to_block_map(self) -> Mapping[BaseSlot, BaseBlock]:
        """Method that creates :attr:`slot_to_block_map`.
        """
        slot_to_block_map: Mapping[BaseSlot, BaseBlock] = dict()
        for block in self.pipeline.nodes:
            for slot in block.slots.inputs.values():
                slot_to_block_map[slot] = block
            for slot in block.slots.outputs.values():
                slot_to_block_map[slot] = block
        return slot_to_block_map

    # contains extra links for pipeline input slots (cyclic)
    def _get_connection_map(self) -> Mapping[BlockInputSlot, BlockOutputSlot]:
        """Method that creates :attr:`conn_dict` and checks if every input slot
        has no more than one corresponding output slot.

        Raises:
            AssertionError: If some input slot has more than one corresponding output slot.
        """
        conn_dict: Mapping[BlockInputSlot, BlockOutputSlot] = dict()
        for conn in self.pipeline.connections:
            assert (conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
            if self._is_input_slot_required(conn.dst):
                conn_dict[conn.dst] = conn.src
        for input_or_inputs in self.inputs.values():
            if isinstance(input_or_inputs, Iterable):
                inputs = input_or_inputs
            else:
                inputs = [input_or_inputs]
            for inp in inputs:
                conn_dict[inp] = inp
        return conn_dict

    def __init__(self, pipeline: BasePipeline, *,
                 stage: None | Stage = None,
                 inputs: None | Mapping[str, BlockInputSlot | Sequence[BlockInputSlot]] = None,
                 outputs: None | Mapping[str, BlockOutputSlot] = None,
                 painter_levels_sep: float = 1.0, figure_dpi: int = 150, figure_rankdir: str = 'LR'):

        super().__init__(pipeline, stage=stage, inputs=inputs, outputs=outputs)

        self.conn_dict = self._get_connection_map()
        self.slot_to_block_map = self._get_slot_to_block_map()
        self.levels_sep = painter_levels_sep
        self.dpi = figure_dpi
        self.rankdir = figure_rankdir

    def _dfs(self, aj_list: Mapping[BaseBlock, BaseBlock], begin_nodes: Sequence[BaseBlock]) -> Set[BaseBlock]:
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

    def _topological_sort(self, aj_list: Mapping[BaseBlock, BaseBlock], begin_nodes: Sequence[BaseBlock]) -> List[BaseBlock]:
        """Method that performs the topological sort of the computational graph.
        The algorithm begins its work from the nodes `begin_nodes`. The algorithm is written using recursive scheme.

        Args:
            aj_list: The graph adjacency list.
            begin_nodes: The graph nodes, which will be used as the algorithm start.
        
        Returns:
            List of the graph nodes in topological order.

        Todo:
            Rewrite the sort with the iterative cheme. Also, think of the performing graph computation during the sort.

        """
        visited: Set[BaseBlock] = set()
        stack: Deque[BaseBlock] = deque()

        def rec_helper(node) -> None:
            visited.add(node)
            for neig_node in aj_list[node]:
                if neig_node not in visited:
                    rec_helper(neig_node)
            stack.append(node)

        for node in begin_nodes:
            if node not in visited:
                rec_helper(node)

        result = list(stack)
        result.reverse()
        return result

    def _get_backward_aj_list(self, feasible_set: None | Set[BaseBlock] = None) -> Mapping[BaseBlock, Set[BaseBlock]]:
        """The internal helper method for making backwards adjacency list 
        (block to block, from the end to the start) of the pipeline.
        
        Args:
            feasible_set: The set of the blocks, which will be used in the adjacency list. 
                Other blocks will not be included. If `None`, all blocks of the pipeline will be used. 

        Returns:
            The backwards adjacency list containing blocks from the `feasible set`.
        """
        backward_aj_list: Mapping[BaseBlock, Set[BaseBlock]] = defaultdict(set)
        for inp_slot, out_slot in self.conn_dict.items():
            if feasible_set is None or \
               self.slot_to_block_map[inp_slot] in feasible_set and self.slot_to_block_map[out_slot] in feasible_set:
                backward_aj_list[self.slot_to_block_map[inp_slot]].add(self.slot_to_block_map[out_slot])
        return backward_aj_list

    def _get_forward_aj_list(self, feasible_set: None | Set[BaseBlock] = None) -> Mapping[BaseBlock, Set[BaseBlock]]:
        """The internal helper method for making adjacency list (block to block) of the pipeline.
        
        Args:
            feasible_set: The set of the blocks, which will be used in the adjacency list. 
                Other blocks will not be included. If `None`, all blocks of the pipeline will be used. 

        Returns:
            The adjacency list containing blocks from the `feasible set`.
        """
        forward_aj_list: Mapping[BaseBlock, Set[BaseBlock]] = defaultdict(set)
        for inp_slot, out_slot in self.conn_dict.items():
            if feasible_set is None or \
               self.slot_to_block_map[inp_slot] in feasible_set and self.slot_to_block_map[out_slot] in feasible_set:
                forward_aj_list[self.slot_to_block_map[out_slot]].add(self.slot_to_block_map[inp_slot])
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

        Todo:
            Think of the partial graph use: now the topology sort is performed using pipeline's inputs, 
            not the inputs in `input_values`.

        """
        self.__check_inputs_concordance(input_values)

        output_blocks = [self.slot_to_block_map[slot] for slot in self.outputs.values()]
        backward_pass = self._dfs(self._get_backward_aj_list(), output_blocks)

        input_blocks = set([self.slot_to_block_map[slot] for slot in self.inputs.values()])
        topological_order = self._topological_sort(self._get_forward_aj_list(backward_pass), input_blocks)

        slots_values: Dict[BaseSlot, Data] = dict()
        for input_name, input_data in input_values.items():
            input_or_inputs = self.inputs[input_name]
            if isinstance(input_or_inputs, Iterable):
                inputs = input_or_inputs
            else:
                inputs = [input_or_inputs]
            for inp in inputs:
                slots_values[inp] = input_data
        for node in topological_order:
            node_input_data = dict()
            for inp_slot in node.slots.inputs.values():
                if inp_slot not in self.conn_dict:
                    # input slot was not used
                    continue
                corresponding_output = self.conn_dict[inp_slot]
                node_input_data[inp_slot] = slots_values[corresponding_output]
            outputs = self._execute_block(node, node_input_data)
            slots_values.update(outputs)
        result: Mapping[str, Data]  = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = slots_values[output_slot]
        return result

    def draw(self, output_filename: str) -> None:
        """Method for the computational graph drawing. This executor uses 
        the graphviz library.

        * The solid black edges and nodes are the ones that will be used during calculations. 
        * The dashed will be skipped during the optimization. 
        * The red colored nodes mean inputs and outputs, the red colored edges signalize that
          type of the block's input slot was misspecified: according to the :attr:`stage` metainformation 
          the connection should be used, but the corresponding block won't be used because of the optimization.

        Args:
            output_filename: Path (containing the filename) where the output graphics file will be saved.
                The filename determines the format, for example, figure.png will be rendered as a raster graphics file
                and figure.pdf - as a vector one. The range of the formats depends on the cairo renderer: https://graphviz.org/docs/outputs/.
        """
        output_blocks = [self.slot_to_block_map[slot] for slot in self.outputs.values()]
        backward_pass = self._dfs(self._get_backward_aj_list(), output_blocks)

        input_blocks = set([self.slot_to_block_map[slot] for slot in self.inputs.values()])
        forward_pass = self._dfs(self._get_forward_aj_list(backward_pass), input_blocks)
        used_blocks = backward_pass & forward_pass

        graph = gv.Digraph('DeepForestGraph', renderer='cairo', formatter='cairo', node_attr={'shape': 'record'})
        graph.attr(rankdir=self.rankdir, ranksep=str(self.levels_sep), dpi=str(self.dpi))

        # drawing blocks
        for block in self.pipeline.nodes:
            inputs = block.slots.inputs
            outputs = block.slots.outputs
            inputs_info = '|'.join([f'<i{hash(slot)}> {name}' for name, slot in inputs.items()])
            outputs_info = '|'.join([f'<o{hash(slot)}> {name}' for name, slot in outputs.items()])
            block_name = block.name if hasattr(block, 'name') else block.__class__.__name__
            node_style = 'dashed' if block not in used_blocks else ''
            graph.node(f'block{id(block)}', f'{block_name}|{{{{{inputs_info}}}|{{{outputs_info}}}}}', style=node_style)

        # drawing edges
        for conn in self.pipeline.connections:
            is_conn_req = self._is_input_slot_required(conn.dst)
            edge_style = 'dashed' if not is_conn_req else ''
            edge_color = 'red' if self.slot_to_block_map[conn.dst] not in used_blocks and is_conn_req else 'black'
            graph.edge(f'block{id(self.slot_to_block_map[conn.src])}:o{hash(conn.src)}',
                   f'block{id(self.slot_to_block_map[conn.dst])}:i{hash(conn.dst)}', style=edge_style, color=edge_color)

        # drawing input slots
        for inp_name, inp_slots in self.inputs.items():
            graph.node(f'inp_{inp_name}', f'<I_{inp_name}> Input "{inp_name}"', color='red')
            f_node_needed = False
            if not isinstance(inp_slots, Iterable):
                inp_slots = [inp_slots]
            for slot in inp_slots:
                if self._is_input_slot_required(slot):
                    edge_style = ''
                    f_node_needed = True
                else:
                    edge_style = 'dashed'
                graph.edge(f'inp_{inp_name}:I_{inp_name}',
                       f'block{id(self.slot_to_block_map[slot])}:i{hash(slot)}', style=edge_style)
            node_style = '' if f_node_needed else 'dashed'
            graph.node(f'inp_{inp_name}', style=node_style)

        # drawing output slots
        for out_name, out_slots in self.outputs.items():
            graph.node(f'out_{out_name}', f'<O_{out_name}> Output "{out_name}"', color='red')
            f_node_needed = False
            if not isinstance(out_slots, Iterable):
                out_slots = [out_slots]
            for slot in out_slots:
                if self._is_input_slot_required(slot):
                    edge_style = ''
                    f_node_needed = True
                else:
                    edge_style = 'dashed'
                graph.edge(f'block{id(self.slot_to_block_map[slot])}:o{hash(slot)}', f'out_{out_name}:O_{out_name}')
            node_style = '' if f_node_needed else 'dashed'
            graph.node(f'out_{out_name}', style=node_style)

        graph.render(outfile=output_filename, cleanup=True)

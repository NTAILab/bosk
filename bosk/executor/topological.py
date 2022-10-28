from typing import Dict, Mapping, Sequence, Union
from collections.abc import Iterable
from collections import defaultdict, deque

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from ..slot import BlockInputSlot, BlockOutputSlot, InputSlotMeta, OutputSlotMeta
from ..pipeline import BasePipeline

import graphviz as gv

class TopologicalExecutor(BaseExecutor):
    def __check_inputs_concordance(self, input_values: Mapping[str, Data]):
        passed = 0
        for inp_name, inp_slot in self.inputs.items():
            assert(inp_name in input_values), f"Unable to find input slot {inp_name} (id {hash(inp_slot)}) in input data"
            passed += 1
        assert(passed == len(input_values)), f"Input values are incompatible with pipeline input slots"
    
    def _get_slot_to_block_map(self):
        slot_to_block_map = dict()
        for block in self.pipeline.nodes:
            for slot in block.slots.inputs.values():
                slot_to_block_map[slot] = block
            for slot in block.slots.outputs.values():
                slot_to_block_map[slot] = block
        return slot_to_block_map
    
    # contains extra links for input slots (cyclic)
    def _get_connection_map(self):
        conn_dict = dict()
        for conn in self.pipeline.connections:
            assert(conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
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
                 inputs: None | Mapping[str, InputSlotMeta | Sequence[InputSlotMeta]] = None,
                 outputs: None | Mapping[str, OutputSlotMeta] = None):
        
        super().__init__(pipeline, stage=stage, inputs=inputs, outputs=outputs)
        
        self.conn_dict = self._get_connection_map()
        self.slot_to_block_map = self._get_slot_to_block_map()
        self.used_blocks = None

    # returns set of the visited nodes
    def _dfs(self, aj_list, begin_nodes):
        visited = set(begin_nodes)
        stack = deque(begin_nodes)
        while stack:
            node = stack.pop()
            for neig_node in aj_list[node]:
                if neig_node not in visited:
                    visited.add(neig_node)
                    stack.append(neig_node)
        return visited

    def _topological_sort(self, aj_list, begin_nodes):
        visited = set()
        stack = deque()
        
        def rec_helper(node):
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

    def _get_backward_aj_list(self, feasible_set=None):
        backward_aj_list = defaultdict(set) # Block to block
        for input, output in self.conn_dict.items():
            if feasible_set is None or \
               self.slot_to_block_map[input] in feasible_set and self.slot_to_block_map[output] in feasible_set:
                backward_aj_list[self.slot_to_block_map[input]].add(self.slot_to_block_map[output])
        return backward_aj_list

    def _get_forward_aj_list(self, feasible_set=None):
        forward_aj_list = defaultdict(set) # Block to block
        for input, output in self.conn_dict.items():
            if feasible_set is None or \
               self.slot_to_block_map[input] in feasible_set and self.slot_to_block_map[output] in feasible_set:
                forward_aj_list[self.slot_to_block_map[output]].add(self.slot_to_block_map[input])
        return forward_aj_list

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        self.__check_inputs_concordance(input_values)

        output_blocks = [self.slot_to_block_map[slot] for slot in self.outputs.values()]
        backward_pass = self._dfs(self._get_backward_aj_list(), output_blocks)
        
        input_blocks = set([self.slot_to_block_map[slot] for slot in self.inputs.values()])
        topological_order = self._topological_sort(self._get_forward_aj_list(backward_pass), input_blocks)

        slots_values: Dict[BlockInputSlot, Data] = dict()
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
        result = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = slots_values[output_slot]
        return result
        
    def draw(self, output_name, levels_sep=1.0, dpi=150):
        output_blocks = [self.slot_to_block_map[slot] for slot in self.outputs.values()]
        backward_pass = self._dfs(self._get_backward_aj_list(), output_blocks)

        input_blocks = set([self.slot_to_block_map[slot] for slot in self.inputs.values()])
        forward_pass = self._dfs(self._get_forward_aj_list(backward_pass), input_blocks)
        used_blocks = backward_pass & forward_pass

        g = gv.Digraph('DeepForestGraph', renderer='cairo', formatter='cairo', node_attr={'shape': 'record'})
        g.attr(rankdir='LR', ranksep=str(levels_sep), dpi=str(dpi))

        # drawing blocks
        for block in self.pipeline.nodes:
            inputs = block.slots.inputs
            outputs = block.slots.outputs
            inputs_info = '|'.join([f'<i{hash(slot)}> {name}' for name, slot in inputs.items()])
            outputs_info = '|'.join([f'<o{hash(slot)}> {name}' for name, slot in outputs.items()])
            block_name = block.name if hasattr(block, 'name') else block.__class__.__name__
            node_style = 'dashed' if block not in used_blocks else ''
            g.node(f'block{id(block)}', f'{block_name}|{{{{{inputs_info}}}|{{{outputs_info}}}}}', style=node_style)
        
        # drawing edges
        for conn in self.pipeline.connections:
            is_conn_req = self._is_input_slot_required(conn.dst)
            edge_style = 'dashed' if not is_conn_req else ''
            edge_color = 'red' if self.slot_to_block_map[conn.dst] not in used_blocks and is_conn_req else 'black'
            g.edge(f'block{id(self.slot_to_block_map[conn.src])}:o{hash(conn.src)}', \
                f'block{id(self.slot_to_block_map[conn.dst])}:i{hash(conn.dst)}', style=edge_style, color=edge_color)
        
        # drawing input slots
        for inp_name, inp_slots in self.inputs.items():
            g.node(f'inp_{inp_name}', f'<I_{inp_name}> Input "{inp_name}"', color='red')
            f_node_needed = False
            if not isinstance(inp_slots, Iterable):
                inp_slots = [inp_slots]
            for slot in inp_slots:
                if self._is_input_slot_required(slot):
                    edge_style = ''
                    f_node_needed = True
                else:
                    edge_style = 'dashed'
                g.edge(f'inp_{inp_name}:I_{inp_name}', f'block{id(self.slot_to_block_map[slot])}:i{hash(slot)}', style=edge_style)
            node_style = '' if f_node_needed else 'dashed'
            g.node(f'inp_{inp_name}', style=node_style)
        
        # drawing output slots
        for out_name, out_slots in self.outputs.items():
            g.node(f'out_{out_name}', f'<O_{out_name}> Output "{out_name}"', color='red')
            f_node_needed = False
            if not isinstance(out_slots, Iterable):
                out_slots = [out_slots]
            for slot in out_slots:
                if self._is_input_slot_required(slot):
                    edge_style = ''
                    f_node_needed = True
                else:
                    edge_style = 'dashed'
                g.edge(f'block{id(self.slot_to_block_map[slot])}:o{hash(slot)}', f'out_{out_name}:O_{out_name}')
            node_style = '' if f_node_needed else 'dashed'
            g.node(f'out_{out_name}', style=node_style)
        
        g.render(outfile=output_name, cleanup=True)
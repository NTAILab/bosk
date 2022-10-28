from typing import Dict, Mapping, Sequence, Union
from collections.abc import Iterable
from collections import defaultdict, deque

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from ..block.base import BaseBlock, BlockInputData, BlockOutputData
from ..slot import BlockInputSlot, BlockOutputSlot

import graphviz as gv

class TopologicalExecutor(BaseExecutor):

    # returns set of the visited nodes
    def __dfs(self, aj_list, begin_nodes):
        visited = set(begin_nodes)
        stack = deque(begin_nodes)
        while stack:
            node = stack.pop()
            for neig_node in aj_list[node]:
                if neig_node not in visited:
                    visited.add(neig_node)
                    stack.append(neig_node)
        return visited

    def __topological_sort(self, aj_list, begin_nodes):
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

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        # need to think how to get access from slot to the parent block without circular links
        inp_slots_to_blocks = dict()
        out_slots_to_blocks = dict()
        for block in self.pipeline.nodes:
            for slot in block.slots.inputs.values():
                inp_slots_to_blocks[slot] = block
            for slot in block.slots.outputs.values():
                out_slots_to_blocks[slot] = block


        # typing.Mapping[bosk.block.base.BlockInputSlot, bosk.block.base.BlockOutputSlot] is not a generic class
        conn_dict = dict() #Mapping[BlockInputSlot, BlockOutputSlot] # Input to output
        backward_aj_list = defaultdict(set) # Block to block
        
        # fill ajacency lists
        for conn in self.pipeline.connections:
            assert(conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
            conn_dict[conn.dst] = conn.src
            backward_aj_list[inp_slots_to_blocks[conn.dst]].add(out_slots_to_blocks[conn.src])

        # check self.inputs and input_data
        passed = 0
        for inp_name, inp_slot in self.inputs.items():
            # assert(inp_slot in conn_dict), f"Unable to find block with input {inp_name} (id {hash(inp_slot)}) in pipeline"
            assert(inp_name in input_values), f"Unable to find input slot {inp_name} (id {hash(inp_slot)}) in input data"
            passed += 1
        assert(passed == len(input_values)), f"Input values are incompatible with pipeline input slots"

        output_blocks = [out_slots_to_blocks[slot] for slot in self.outputs.values()]
        backward_pass = self.__dfs(backward_aj_list, output_blocks)
        used_blocks = backward_pass
        aj_list = defaultdict(set) # Block to block
        for conn in self.pipeline.connections:
            if out_slots_to_blocks[conn.src] in used_blocks and inp_slots_to_blocks[conn.dst] in used_blocks:
                aj_list[out_slots_to_blocks[conn.src]].add(inp_slots_to_blocks[conn.dst])

        input_blocks = set([inp_slots_to_blocks[slot] for slot in self.inputs.values()])
        topological_order = self.__topological_sort(aj_list, input_blocks)

        slots_values: Dict[BlockInputSlot, Data] = dict()
        for input_name, input_data in input_values.items():
            input_or_inputs = self.inputs[input_name]
            if isinstance(input_or_inputs, Sequence):
                inputs = input_or_inputs
            else:
                inputs = [input_or_inputs]
            for inp in inputs:
                slots_values[inp] = input_data
                conn_dict[inp] = inp 
        for node in topological_order:
            #
            # print(node.__class__.__name__)
            #
            node_input_data = dict()
            for inp_slot in node.slots.inputs.values():
                # what is that, optimization? next if should solve it
                if not self._is_input_slot_required(inp_slot):
                    continue
                if inp_slot not in conn_dict:
                    # print("What does it mean?")
                    continue
                corresponding_output = conn_dict[inp_slot]
                node_input_data[inp_slot] = slots_values[corresponding_output]
            outputs = self._execute_block(node, node_input_data)
            # inputs = dict()
            slots_values.update(outputs)
            # slots_values.update(inputs)
        result = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = slots_values[output_slot]
        return result
        
    def draw(self, output_name, levels_sep=1.0, dpi=150):
        inp_slots_to_blocks = dict()
        out_slots_to_blocks = dict()
        for block in self.pipeline.nodes:
            for slot in block.slots.inputs.values():
                inp_slots_to_blocks[slot] = block
            for slot in block.slots.outputs.values():
                out_slots_to_blocks[slot] = block
        
        conn_dict = dict() #Mapping[BlockInputSlot, BlockOutputSlot] # Input to output
        backward_aj_list = defaultdict(set) # Block to block
        
        # will be rewritten after discussion
        for conn in self.pipeline.connections:
            assert(conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
            conn_dict[conn.dst] = conn.src
            backward_aj_list[inp_slots_to_blocks[conn.dst]].add(out_slots_to_blocks[conn.src])


        output_blocks = [out_slots_to_blocks[slot] for slot in self.outputs.values()]
        backward_pass = self.__dfs(backward_aj_list, output_blocks)
        
        aj_list = defaultdict(set) # Block to block
        for conn in self.pipeline.connections:
            if out_slots_to_blocks[conn.src] in backward_pass and inp_slots_to_blocks[conn.dst] in backward_pass:
                aj_list[out_slots_to_blocks[conn.src]].add(inp_slots_to_blocks[conn.dst])
        input_blocks = set([inp_slots_to_blocks[slot] for slot in self.inputs.values()])
        forward_pass = self.__dfs(aj_list, input_blocks)
        used_blocks = backward_pass & forward_pass


        g = gv.Digraph('DeepForestGraph', renderer='cairo', formatter='cairo', node_attr={'shape': 'record'})
        g.attr(rankdir='LR', ranksep=str(levels_sep), dpi=str(dpi))

        for block in self.pipeline.nodes:
            inputs = block.slots.inputs
            outputs = block.slots.outputs
            inputs_info = '|'.join([f'<i{hash(slot)}> {name}' for name, slot in inputs.items()])
            outputs_info = '|'.join([f'<o{hash(slot)}> {name}' for name, slot in outputs.items()])
            block_name = block.name if hasattr(block, 'name') else block.__class__.__name__
            node_style = 'dashed' if block not in used_blocks else ''
            g.node(f'block{id(block)}', f'{block_name}|{{{{{inputs_info}}}|{{{outputs_info}}}}}', style=node_style)
        for inp_slot, out_slot in conn_dict.items():
            edge_style = 'dashed' if not self._is_input_slot_required(inp_slot) else ''
            g.edge(f'block{id(out_slots_to_blocks[out_slot])}:o{hash(out_slot)}', f'block{id(inp_slots_to_blocks[inp_slot])}:i{hash(inp_slot)}', style=edge_style)
        
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
                g.edge(f'inp_{inp_name}:I_{inp_name}', f'block{id(inp_slots_to_blocks[slot])}:i{hash(slot)}', style=edge_style)
            node_style = '' if f_node_needed else 'dashed'
            g.node(f'inp_{inp_name}', style=node_style)
            
        
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
                g.edge(f'block{id(out_slots_to_blocks[slot])}:o{hash(slot)}', f'out_{out_name}:O_{out_name}')
            node_style = '' if f_node_needed else 'dashed'
            g.node(f'out_{out_name}', style=node_style)
        
        g.render(outfile=output_name, cleanup=True)
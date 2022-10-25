from typing import Dict, Mapping, Sequence, Union
from collections import defaultdict, deque
from collections.abc import Iterable

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from ..block.base import BaseBlock, BlockInputSlot, BlockOutputSlot

import graphviz as gv

class PipelinePainter():
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
        # stack = deque()
        order_by_levels = defaultdict(list)
        
        def rec_helper(node, level=1):
            visited.add(node)
            for neig_node in aj_list[node]:
                if neig_node not in visited:
                    rec_helper(neig_node, level + 1)
            order_by_levels[level].append(node)
        
        for node in begin_nodes:
            if node not in visited:
                rec_helper(node)
        
        return order_by_levels

    def __init__(self, pipeline, inputs: None | Mapping[str, BlockInputSlot | Sequence[BlockInputSlot]] = None,
                 outputs: None | Mapping[str, BlockOutputSlot] = None):
        self.pipeline = pipeline
        self.inputs = inputs
        self.outputs = outputs
    
    def draw(self):
        conn_dict = dict() #Mapping[BlockInputSlot, BlockOutputSlot] # Input to output
        backward_aj_list = defaultdict(set) # Block to block
        
        # fill ajacency lists
        for conn in self.pipeline.connections:
            assert(conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
            conn_dict[conn.dst] = conn.src
            backward_aj_list[conn.dst.parent_block].add(conn.src.parent_block)

        # output_blocks = [slot.parent_block for slot in self.outputs.values()]
        # backward_pass = self.__dfs(backward_aj_list, output_blocks)
        # used_blocks = backward_pass# & forward_pass
        # aj_list = defaultdict(set) # Block to block
        # for conn in self.pipeline.connections:
        #     if conn.src.parent_block in used_blocks and conn.dst.parent_block in used_blocks:
        #         aj_list[conn.src.parent_block].add(conn.dst.parent_block)
        # # needs to be discussed! (11)
        # input_blocks = set([slot.parent_block for slot in self.inputs.values()])
        # topological_order = self.__topological_sort(aj_list, input_blocks)
        g = gv.Digraph('DeepForestGraph', filename='DeepForestGraph',
                     node_attr={'shape': 'record'})
        g.attr(rankdir='LR')

        for block in self.pipeline.nodes:
            inputs = block.meta.inputs
            outputs = block.meta.outputs
            inputs_info = '|'.join([f'<i{hash(slot)}> {name}' for name, slot in inputs.items()])
            outputs_info = '|'.join([f'<o{hash(slot)}> {name}' for name, slot in outputs.items()])
            block_name = block.name if hasattr(block, 'name') else block.__class__.__name__
            g.node(f'block{id(block)}', f'{block_name}|{{{{{inputs_info}}}|{{{outputs_info}}}}}')
        for inSlot, outSlot in conn_dict.items():
            g.edge(f'block{id(outSlot.parent_block)}:o{hash(outSlot)}', f'block{id(inSlot.parent_block)}:i{hash(inSlot)}')
        
        for inp_name, inp_slots in self.inputs.items():
            g.node(f'inp_{inp_name}', f'<I_{inp_name}> Input "{inp_name}"', color='red')
            if not isinstance(inp_slots, Iterable):
                inp_slots = [inp_slots]
            for slot in inp_slots:
                g.edge(f'inp_{inp_name}:I_{inp_name}', f'block{id(slot.parent_block)}:i{hash(slot)}')
        
        for out_name, out_slots in self.outputs.items():
            g.node(f'out_{out_name}', f'<O_{out_name}> Output "{out_name}"', color='red')
            if not isinstance(out_slots, Iterable):
                out_slots = [out_slots]
            for slot in out_slots:
                g.edge(f'block{id(slot.parent_block)}:o{hash(slot)}', f'out_{out_name}:O_{out_name}')

        g.view()



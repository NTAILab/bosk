from typing import Dict, Mapping, Sequence, Union
from collections import defaultdict, deque

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from ..block.base import BaseBlock, BlockInputSlot, BlockOutputSlot, BaseSlot
# from ..block.slot import BlockInputSlot, BlockOutputSlot, BaseSlot


InputSlotToDataMapping = Mapping[BlockInputSlot, Data]
"""Block input slot data mapping.

It is indexed by input slots.
"""


def execute_block(executor_stage: Stage, node: BaseBlock, node_input_mapping: InputSlotToDataMapping):
        if executor_stage == Stage.FIT:
            node.fit({
                slot.name: values
                for slot, values in node_input_mapping.items()
                if slot.stages.fit
            })
        filtered_node_input_mapping = {
            slot.name: values
            for slot, values in node_input_mapping.items()
            if slot.stages.transform or (executor_stage == Stage.FIT and slot.stages.transform_on_fit)
        }
        return node.wrap(node.transform(filtered_node_input_mapping))

def is_input_slot_required(executor_stage: Stage, input_slot: BlockInputSlot) -> bool:
        if executor_stage == Stage.FIT:
            return input_slot.stages.fit \
                or input_slot.stages.transform \
                or input_slot.stages.transform_on_fit
        elif executor_stage == Stage.TRANSFORM:
            return input_slot.stages.transform
        else:
            raise NotImplementedError()

class NaiveExecutor(BaseExecutor):
    """Naive executor implementation.

    Considers only input-output meta information to match slots.

    """

    def __is_input_slot_required(self, input_slot: BlockInputSlot) -> bool:
        if self.stage == Stage.FIT:
            return input_slot.stages.fit \
                or input_slot.stages.transform \
                or input_slot.stages.transform_on_fit
        elif self.stage == Stage.TRANSFORM:
            return input_slot.stages.transform
        else:
            raise NotImplementedError()

    def __execute_block(self, node: BaseBlock, node_input_mapping: InputSlotToDataMapping):
        if self.stage == Stage.FIT:
            node.fit({
                slot.name: values
                for slot, values in node_input_mapping.items()
                if slot.stages.fit
            })
        filtered_node_input_mapping = {
            slot.name: values
            for slot, values in node_input_mapping.items()
            if slot.stages.transform or (self.stage == Stage.FIT and slot.stages.transform_on_fit)
        }
        return node.wrap(node.transform(filtered_node_input_mapping))

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        slots_values: Dict[Union[BlockInputSlot, BlockOutputSlot], Data] = dict()
        # fill slots values
        for input_name, input_data in input_values.items():
            input_or_inputs = self.inputs[input_name]
            if isinstance(input_or_inputs, Sequence):
                inputs = input_or_inputs
            else:
                inputs = [input_or_inputs]
            for inp in inputs:
                slots_values[inp] = input_data

        # recursively compute outputs

        def _compute_output(out_slot: BlockOutputSlot):
            assert isinstance(out_slot, BlockOutputSlot)
            if out_slot in slots_values:
                return slots_values[out_slot]
            # search for node which computes the slot value
            for node in self.pipeline.nodes:
                if out_slot not in node.meta.outputs.values():
                    continue
                break
            else:
                raise RuntimeError(f'Node that can compute value for the slot not found: {out_slot}')
            # print("Computing output for node", node._AutoBlock__instance.name, node)

            node_input_slots = node.meta.inputs.values()
            node_input_mapping = dict()
            for input in node_input_slots:
                if not self.__is_input_slot_required(input):
                    continue
                if input in slots_values:
                    node_input_mapping[input] = slots_values[input]
                    continue

                connections = self.pipeline.find_connections(dst=input)
                if len(connections) == 0:
                    continue
                assert len(connections) == 1, f'len(connections) == {len(connections)} for dst {input}'
                conn = connections[0]
                conn_value = _compute_output(conn.src)
                slots_values[conn.src] = conn_value
                slots_values[conn.dst] = conn_value # why?
                node_input_mapping[input] = conn_value

            outputs = self.__execute_block(node, node_input_mapping)
            slots_values.update(outputs)
            return slots_values[out_slot]

        result = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = _compute_output(output_slot)
        return result


class LessNaiveExecutor(BaseExecutor):

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
        # typing.Mapping[bosk.block.base.BlockInputSlot, bosk.block.base.BlockOutputSlot] is not a generic class
        conn_dict = dict() #Mapping[BlockInputSlot, BlockOutputSlot] # Input to output
        backward_aj_list = defaultdict(set) # Block to block
        
        # fill ajacency lists
        for conn in self.pipeline.connections:
            assert(conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
            conn_dict[conn.dst] = conn.src
            backward_aj_list[conn.dst.parent_block].add(conn.src.parent_block)

        # check self.inputs and input_data
        passed = 0
        for inp_name, inp_slot in self.inputs.items():
            # assert(inp_slot in conn_dict), f"Unable to find block with input {inp_name} (id {hash(inp_slot)}) in pipeline"
            assert(inp_name in input_values), f"Unable to find input slot {inp_name} (id {hash(inp_slot)}) in input data"
            passed += 1
        assert(passed == len(input_values)), f"Input values are incompatible with pipeline input slots"

        output_blocks = [slot.parent_block for slot in self.outputs.values()]
        backward_pass = self.__dfs(backward_aj_list, output_blocks)
        # forward_pass = self.__dfs(forward_aj_list, self.inputs)
        # if len(forward_pass) != len(backward_pass): # covers not all cases!
            # print("Warning: some blocks are not used in the pipeline")
        used_blocks = backward_pass# & forward_pass
        aj_list = defaultdict(set) # Block to block
        for conn in self.pipeline.connections:
            if conn.src.parent_block in used_blocks and conn.dst.parent_block in used_blocks:
                aj_list[conn.src.parent_block].add(conn.dst.parent_block)
        # needs to be discussed! (11)
        # need to intersect with used_blocks?
        input_blocks = set([slot.parent_block for slot in self.inputs.values()])
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
                conn_dict[inp] = inp # kostil for input blocks
        for node in topological_order:
            #
            print(node.__class__.__name__)
            #
            node_input_data = dict()
            for inp_slot in node.meta.inputs.values():
                # what is that, optimization? next if should solve it
                if not is_input_slot_required(self.stage, inp_slot):
                    continue
                if inp_slot not in conn_dict:
                    # print("What does it mean?")
                    continue
                corresponding_output = conn_dict[inp_slot]
                node_input_data[inp_slot] = slots_values[corresponding_output]
            outputs = execute_block(self.stage, node, node_input_data)
            # inputs = dict()
            slots_values.update(outputs)
            # slots_values.update(inputs)
        result = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = slots_values[output_slot]
        return result
        







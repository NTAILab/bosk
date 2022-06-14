from typing import Mapping, Sequence

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from ..block.base import BaseBlock, BlockInputData, BlockOutputData


class NaiveExecutor(BaseExecutor):
    """Naive executor implementation.

    Considers only input-output meta information to match slots.

    """

    def __execute_block(self, node: BaseBlock, node_input_mapping: BlockInputData) -> BlockOutputData:
        if self.stage == Stage.FIT:
            node.fit(node_input_mapping)
        filtered_node_input_mapping = {
            slot: values
            for slot, values in node_input_mapping.items()
            if slot.stages.transform
        }
        return node.transform(filtered_node_input_mapping)

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        slots_values = dict()
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

        def _compute_output(out_slot):
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
                slots_values[conn.dst] = conn_value
                node_input_mapping[input] = conn_value
            
            outputs = self.__execute_block(node, node_input_mapping)
            slots_values.update(outputs)
            return slots_values[out_slot]

        result = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = _compute_output(output_slot)
        return result

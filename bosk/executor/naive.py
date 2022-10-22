from typing import Dict, Mapping, Sequence, Union

from ..stages import Stage
from ..data import Data
from .base import BaseExecutor
from ..block.base import BaseBlock, BlockInputData, BlockOutputData
from ..slot import BlockInputSlot, BlockOutputSlot


InputSlotToDataMapping = Mapping[BlockInputSlot, Data]
"""Block input slot data mapping.

It is indexed by input slots.
"""


class NaiveExecutor(BaseExecutor):
    """Naive executor implementation.

    Considers only input-output slots information to match slots.

    """

    def __is_input_slot_required(self, input_slot: BlockInputSlot) -> bool:
        if self.stage == Stage.FIT:
            return input_slot.meta.stages.fit \
                or input_slot.meta.stages.transform \
                or input_slot.meta.stages.transform_on_fit
        elif self.stage == Stage.TRANSFORM:
            return input_slot.meta.stages.transform
        else:
            raise NotImplementedError()

    def __execute_block(self, node: BaseBlock, node_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        if self.stage == Stage.FIT:
            node.fit({
                slot.meta.name: values
                for slot, values in node_input_mapping.items()
                if slot.meta.stages.fit
            })
        filtered_node_input_mapping = {
            slot.meta.name: values
            for slot, values in node_input_mapping.items()
            if slot.meta.stages.transform or (self.stage == Stage.FIT and slot.meta.stages.transform_on_fit)
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
                if out_slot not in node.slots.outputs.values():
                    continue
                break
            else:
                raise RuntimeError(f'Node that can compute value for the slot not found: {out_slot}')
            # print("Computing output for node", node._AutoBlock__instance.name, node)

            node_input_slots = node.slots.inputs.values()
            node_input_mapping = dict()
            for _input in node_input_slots:
                if not self.__is_input_slot_required(_input):
                    continue
                if _input in slots_values:
                    node_input_mapping[_input] = slots_values[_input]
                    continue

                connections = self.pipeline.find_connections(dst=_input)
                if len(connections) == 0:
                    continue
                assert len(connections) == 1, f'len(connections) == {len(connections)} for dst {_input}'
                conn = connections[0]
                conn_value = _compute_output(conn.src)
                slots_values[conn.src] = conn_value
                slots_values[conn.dst] = conn_value
                node_input_mapping[_input] = conn_value

            outputs = self.__execute_block(node, node_input_mapping)
            slots_values.update(outputs)
            return slots_values[out_slot]

        result = dict()
        for output_name, output_slot in self.outputs.items():
            result[output_name] = _compute_output(output_slot)
        return result

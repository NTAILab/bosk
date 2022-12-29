from typing import Dict, Mapping, Union, Sequence, Optional

from ..data import Data
from .base import BaseExecutor
from ..pipeline import BasePipeline
from .descriptor import HandlingDescriptor
from ..block.slot import BlockInputSlot, BlockOutputSlot
from .utility import get_connection_map


class RecursiveExecutor(BaseExecutor):
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

    def __init__(self, pipeline: BasePipeline, handl_desc: HandlingDescriptor,
                inputs: Optional[Sequence[str]] = None, outputs: Optional[Sequence[str]] = None) -> None:
        super().__init__(pipeline, handl_desc, inputs, outputs)
        self._conn_map = get_connection_map(self)

    def __call__(self, input_values: Mapping[str, Data]) -> Mapping[str, Data]:
        self._check_input_values(input_values)
        slots_values: Dict[Union[BlockInputSlot, BlockOutputSlot], Data] = dict()
        slots_values.update(self._map_input_names_to_slots(input_values))

        # recursively compute outputs
        def _compute_output(out_slot: BlockOutputSlot):
            assert isinstance(out_slot, BlockOutputSlot)
            if out_slot in slots_values:
                return slots_values[out_slot]
            
            node = out_slot.parent_block
            node_input_slots = node.slots.inputs.values()
            node_input_mapping = dict()
            for _input in node_input_slots:
                if _input in slots_values:
                    node_input_mapping[_input] = slots_values[_input]
                    continue
                
                _output = self._conn_map.get(_input, None)
                if _output is None:
                    continue

                conn_value = _compute_output(_output)
                slots_values[_output] = conn_value
                slots_values[_input] = conn_value
                node_input_mapping[_input] = conn_value

            outputs = self._execute_block(node, node_input_mapping)
            slots_values.update(outputs)
            return slots_values[out_slot]

        result = dict()
        for output_name, output_slot in self.pipeline.outputs.items():
            if self.outputs is None or output_name in self.outputs:
                result[output_name] = _compute_output(output_slot)
        return result

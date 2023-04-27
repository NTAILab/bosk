from typing import Dict, Mapping, Union, Sequence, Optional


from ..data import BaseData, Data
from .base import BaseBlockExecutor, BaseExecutor, BaseSlotHandler
from .base import Stage
from ..pipeline import BasePipeline
from ..block.base import BlockInputSlot, BlockOutputSlot
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
        stage: Sets :attr:`.BaseExecutor.__stage`,
        inputs: Sets :attr:`.BaseExecutor.__inputs`.
        outputs: Sets :attr:`.BaseExecutor.__outputs`.
        slot_handler: Sets :attr:`.BaseExecutor.__slot_handler` with `_prepare_slot_handler` method.
        block_executor: Sets :attr:`.BaseExecutor.__block_executor` with `_prepare_block_executor` method.
    """

    _conn_map: Mapping[BlockInputSlot, BlockOutputSlot]

    def __init__(self, pipeline: BasePipeline,
                 stage: Stage,
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None,
                 slot_handler: Optional[BaseSlotHandler] = None,
                 block_executor: Optional[BaseBlockExecutor] = None) -> None:
        super().__init__(pipeline, stage, inputs, outputs, slot_handler, block_executor)
        self._conn_map = get_connection_map(self)

    def __call__(self, input_values: Mapping[str, Data]) -> Dict[str, BaseData]:
        self._check_input_values(input_values)
        slots_values: Dict[Union[BlockInputSlot, BlockOutputSlot], Data] = dict()
        # here typing is ignored, because keys in result of `_map_input_names_to_slots`
        # are obviously of subtype of Union[...]
        slots_values.update(self._map_input_names_to_slots(input_values))  # type: ignore

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
            slots_values.update(outputs)  # type: ignore
            return slots_values[out_slot]

        result = dict()
        for output_name, output_slot in self.pipeline.outputs.items():
            if self.outputs is None or output_name in self.outputs:
                result[output_name] = _compute_output(output_slot)
        return result

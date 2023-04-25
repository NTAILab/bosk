"""Universal Pipeline Serializer parametrized by Block Serializer.
"""
from typing import Dict
from ...block.base import BaseBlock, BaseSlot, BlockInputSlot
from .base import BaseBlockSerializer, BasePipelineSerializer
from ..base import BasePipeline
from ..connection import Connection
from zipfile import ZipFile
from json import loads as json_loads, dumps as json_dumps


STRUCTURE_FILENAME = 'structure.json'


class SlotSerializer:
    def __init__(self, ids_by_block=None, blocks_by_id=None):
        self.ids_by_block = ids_by_block
        self.blocks_by_id = blocks_by_id

    def from_dict(self, dictionary: dict) -> BaseSlot:
        parent_block = self.blocks_by_id[dictionary['block_id']]
        name = dictionary['name']
        if parent_block.slots is None:
            # make slots if they were not loaded correctly
            parent_block.slots = parent_block._make_slots()
        if dictionary['is_input']:  # input slot
            for slot_name, slot in parent_block.slots.inputs.items():
                if slot_name == name:
                    return slot
        else:  # output slot
            for slot_name, slot in parent_block.slots.outputs.items():
                if slot_name == name:
                    return slot
        raise ValueError(f'Cannon find the mentioned slot: {dictionary!r}')

    def to_dict(self, slot: BaseSlot) -> dict:
        return dict(
            block_id=self.ids_by_block[slot.parent_block],
            name=slot.meta.name,
            is_input=isinstance(slot, BlockInputSlot),
        )


class ConnectionSerializer:
    def __init__(self, slot_serializer: SlotSerializer):
        self.slot_serializer = slot_serializer

    def from_dict(self, dictionary: dict) -> Connection:
        return Connection(
            src=self.slot_serializer.from_dict(dictionary['src']),
            dst=self.slot_serializer.from_dict(dictionary['dst']),
        )

    def to_dict(self, connection: Connection) -> dict:
        return dict(
            src=self.slot_serializer.to_dict(connection.src),
            dst=self.slot_serializer.to_dict(connection.dst),
        )


class ZipPipelineSerializer(BasePipelineSerializer):
    def __init__(self, block_serializer: BaseBlockSerializer):
        self.block_serializer = block_serializer

    def _dump_block(self, block: BaseBlock, out_file):
        block_slots = block.slots
        block.slots = None
        self.block_serializer.dump(block, out_file)
        block.slots = block_slots

    def _dump_all_blocks(self, ids_by_block, storage):
        for block, block_id in ids_by_block.items():
            with storage.open(f'{block_id}', 'w') as block_outf:
                self._dump_block(block, block_outf)

    def _load_all_blocks(self, storage, block_names) -> Dict[str, BaseBlock]:
        blocks = dict()
        for block_id in block_names:
            with storage.open(f'{block_id}', 'r') as block_inf:
                blocks[block_id] = self.block_serializer.load(block_inf)
        return blocks

    def dump(self, pipeline: BasePipeline, out_file):
        with ZipFile(out_file, 'w') as storage:
            # serialize all blocks
            ids_by_block = {
                block:
                f'block/{i}'
                for i, block in enumerate(pipeline.nodes)
            }
            self._dump_all_blocks(ids_by_block, storage)
            # serialize connections
            slot_serializer = SlotSerializer(ids_by_block=ids_by_block)
            conn_serializer = ConnectionSerializer(slot_serializer)
            serialized_connections = []
            for conn in pipeline.connections:
                serialized_connections.append(conn_serializer.to_dict(conn))
            # serialize inputs, outputs
            serialized_inputs = {
                name: slot_serializer.to_dict(slot)
                for name, slot in pipeline.inputs.items()
            }
            serialized_outputs = {
                name: slot_serializer.to_dict(slot)
                for name, slot in pipeline.outputs.items()
            }
            structure = dict(
                inputs=serialized_inputs,
                outputs=serialized_outputs,
                connections=serialized_connections,
                block_ids=list(ids_by_block.values()),
            )
            with storage.open(STRUCTURE_FILENAME, 'w') as structure_outf:
                structure_outf.write(json_dumps(structure).encode('utf-8'))


    def load(self, in_file) -> BasePipeline:
        with ZipFile(in_file, 'r') as storage:
            with storage.open(STRUCTURE_FILENAME, 'r') as structure_inf:
                structure = json_loads(structure_inf.read())
            blocks_by_id = self._load_all_blocks(storage, structure['block_ids'])
            nodes = list(blocks_by_id.values())
            slot_serializer = SlotSerializer(blocks_by_id=blocks_by_id)
            connection_serializer = ConnectionSerializer(slot_serializer)
            connections = list(map(connection_serializer.from_dict, structure['connections']))
            inputs = {
                name: slot_serializer.from_dict(value)
                for name, value in structure['inputs'].items()
            }
            outputs = {
                name: slot_serializer.from_dict(value)
                for name, value in structure['outputs'].items()
            }
        pipeline = BasePipeline(nodes, connections, inputs, outputs)
        return pipeline

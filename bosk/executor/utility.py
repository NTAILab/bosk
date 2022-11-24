"""Module containing useful utility functions, which
can be used by many executors.
"""

from typing import Mapping
from .base import BaseExecutor
from ..slot import BlockInputSlot, BlockOutputSlot


def get_connection_map(executor: BaseExecutor) -> Mapping[BlockInputSlot, BlockOutputSlot]:
    """Method that creates :attr:`conn_dict` and checks if every input slot
    has no more than one corresponding output slot.

    Raises:
        AssertionError: If some input slot has more than one corresponding output slot.
    """
    conn_dict: Mapping[BlockInputSlot, BlockOutputSlot] = dict()
    for conn in executor.pipeline.connections:
        assert (conn.dst not in conn_dict), f'Input slot of block "{conn.dst.parent_block.__class__.__name__}" \
            (id {hash(conn.dst)}) is used more than once'
        if executor.slot_handler.is_slot_required(conn.dst):
            conn_dict[conn.dst] = conn.src
    return conn_dict
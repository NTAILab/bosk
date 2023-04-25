"""Module containing useful utility functions, which
can be used by many executors.
"""

from typing import Dict, Mapping
from .base import BaseExecutor
from ..block.base import BlockInputSlot, BlockOutputSlot


def get_connection_map(executor: BaseExecutor) -> Mapping[BlockInputSlot, BlockOutputSlot]:
    """Method that creates :attr:`conn_dict` and checks if every input slot
    has no more than one corresponding output slot.

    Raises:
        AssertionError: If some input slot has more than one corresponding output slot.
    """
    conn_dict: Dict[BlockInputSlot, BlockOutputSlot] = dict()
    for conn in executor.pipeline.connections:
        assert (conn.dst not in conn_dict), f'Input slot "{repr(conn.dst)}" of block \
            "{repr(conn.dst.parent_block)}" is used more than once'
        if executor._is_slot_required(conn.dst):
            conn_dict[conn.dst] = conn.src
    return conn_dict

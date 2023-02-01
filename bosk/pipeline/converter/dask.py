from functools import singledispatchmethod
from typing import Any, Dict
from ...visitor.base import BaseVisitor
from ..base import BasePipeline
from ...block.base import BaseBlock
from ..connection import Connection
import dask


class DaskOperators:
    @staticmethod
    def bypass(value):
        return value

    @staticmethod
    def extract(block_output, _output_key: str = None):
        raise NotImplementedError()
        return None

    @staticmethod
    def compute(*inputs, _block=None):
        raise NotImplementedError()
        return {}


class DaskConverter:
    """Converts a pipeline to a Dask computational graph.

    Since Dask does not support multi-output blocks by default,
    we split each block into multiple nodes:
        - Node that computes the block;
        - A number of nodes that extract data corresponding to outputs.
    """

    class Visitor(BaseVisitor):
        def __init__(self, parent):
            self.parent = parent

        @singledispatchmethod
        def visit(self, obj):
            pass  # ignore extra entities

        @visit.register
        def _(self, block: BaseBlock):
            """Enumerate blocks.
            """
            if block in self.parent.block_ids:
                return
            self.parent.block_ids[block] = len(self.parent.block_ids)
            block_mangle = self.parent._mangle_block(block)

            input_mangles = []
            for input_key, input_slot in block.slots.inputs.items():
                in_mangle = self.parent._mangle_input_slot(input_slot)
                input_mangles.append(in_mangle)

            self.parent.dsk[block_mangle] = (
                dask.utils.apply, DaskOperators.compute,
                tuple(sorted(input_mangles)),
                {
                    '_block': block
                }
            )

            for out_key, output_slot in block.slots.outputs.items():
                out_mangle = self.parent._mangle_output_slot(output_slot)
                self.parent.dsk[out_mangle] = (
                    dask.utils.apply, DaskOperators.extract,
                    (block_mangle,),
                    {
                        '_output_key': out_key
                    }
                )


        @visit.register
        def _(self, connection: Connection):
            connection.src.parent_block
            connection.src.meta.name
            in_mangle = self.parent._mangle_input_slot(connection.dst)
            out_mangle = self.parent._mangle_output_slot(connection.src)
            self.parent.dsk[in_mangle] = (DaskOperators.bypass, out_mangle)

        @visit.register
        def _(self, pipeline: BasePipeline):
            pass

    def __init__(self):
        self.dsk = dict()
        self.block_ids = dict()
        self.visitor = self.Visitor(self)

    def _mangle_block(self, block: BaseBlock) -> str:
        return f'block_{self.block_ids[block]}'

    def _mangle_output_slot(self, output_slot) -> str:
        return self._mangle_block(output_slot.parent_block) + '_' + output_slot.meta.name + '_out'

    def _mangle_input_slot(self, input_slot) -> str:
        return self._mangle_block(input_slot.parent_block) + '_' + input_slot.meta.name + '_in'

    def __call__(self, pipeline: BasePipeline) -> Dict[str, Any]:
        pipeline.accept(self.visitor)
        return self.dsk

from functools import singledispatchmethod
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, MutableMapping
from abc import ABC, abstractmethod

from ...stages import Stage
from ...visitor.base import BaseVisitor
from ..base import BasePipeline
from ...block.base import BaseBlock, BlockInputData
from ..connection import Connection
from ...data import BaseData
import dask


class DaskOperatorSet(ABC):
    @staticmethod
    @abstractmethod
    def bypass(value: BaseData) -> BaseData:
        ...

    @staticmethod
    @abstractmethod
    def extract(block_output: Dict[str, BaseData], _output_key: Optional[str] = None):
        ...

    @staticmethod
    @abstractmethod
    def compute(*inputs, _block: Optional[BaseBlock] = None, _input_keys: Optional[List[str]] = None):
        ...


class TransformDaskOperatorSet(DaskOperatorSet):
    @staticmethod
    def bypass(value: BaseData) -> BaseData:
        return value

    @staticmethod
    def extract(block_output: Dict[str, BaseData], _output_key: Optional[str] = None):
        assert _output_key is not None
        return block_output[_output_key]

    @staticmethod
    def compute(*inputs, _block: Optional[BaseBlock] = None,
                _input_keys: Optional[List[str]] = None):
        assert _block is not None
        assert _input_keys is not None
        input_mapping = dict(zip(_input_keys, inputs))
        return _block.transform({
            k: v
            for k, v in input_mapping.items()
            if not isinstance(v, str) and _block.meta.inputs[k].stages.transform
        })


class FitDaskOperatorSet(DaskOperatorSet):
    @staticmethod
    def bypass(value: BaseData) -> BaseData:
        return value

    @staticmethod
    def extract(block_output: Dict[str, BaseData], _output_key: Optional[str] = None):
        assert _output_key is not None
        return block_output[_output_key]

    @staticmethod
    def compute(*inputs, _block: Optional[BaseBlock] = None,
                _input_keys: Optional[List[str]] = None):
        assert _block is not None
        assert _input_keys is not None
        input_mapping = dict(zip(_input_keys, inputs))
        fit_args = {
            k: v
            for k, v in input_mapping.items()
            if not isinstance(v, str) and _block.meta.inputs[k].stages.fit
        }
        _block.fit(fit_args)
        transform_args = {
            k: v
            for k, v in input_mapping.items()
            if not isinstance(v, str) and (
                _block.meta.inputs[k].stages.transform or _block.meta.inputs[k].stages.transform_on_fit
            )
        }
        return _block.transform(transform_args)


class DaskConverter:
    """Converts a pipeline to a Dask computational graph.

    Since Dask does not support multi-output blocks by default,
    we split each block into multiple nodes:

        - Node that computes the block;
        - A number of nodes that extract data corresponding to outputs.

    """

    class Visitor(BaseVisitor):
        def __init__(self, parent: 'DaskConverter'):
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
            input_keys = []
            for input_key, input_slot in block.slots.inputs.items():
                if self.parent.stage == Stage.TRANSFORM:
                    if not input_slot.meta.stages.transform:
                        continue
                elif self.parent.stage == Stage.FIT:
                    if not input_slot.meta.stages.transform and not input_slot.meta.stages.fit \
                            and input_slot.meta.stages.transform_on_fit:
                        continue
                else:
                    raise NotImplementedError('Unhandled stage')

                in_mangle = self.parent._mangle_input_slot(input_slot)
                input_mangles.append(in_mangle)
                input_keys.append(input_key)

            block_args_mangle = f'#args_{block_mangle}'
            self.parent.dsk[block_args_mangle] = input_mangles
            self.parent.dsk[block_mangle] = (
                dask.utils.apply, self.parent.operator_set.compute,
                block_args_mangle,
                {
                    '_block': block,
                    '_input_keys': input_keys,
                }
            )

            for out_key, output_slot in block.slots.outputs.items():
                out_mangle = self.parent._mangle_output_slot(output_slot)
                out_args_mangle = f'#args_{out_mangle}'
                self.parent.dsk[out_args_mangle] = [block_mangle]
                self.parent.dsk[out_mangle] = (
                    dask.utils.apply, self.parent.operator_set.extract,
                    out_args_mangle,
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
            self.parent.dsk[in_mangle] = (self.parent.operator_set.bypass, out_mangle)

        @visit.register
        def _(self, pipeline: BasePipeline):
            # add connections for inputs
            for input_key, input_slot in pipeline.inputs.items():
                self.parent.dsk[self.parent._mangle_input_slot(input_slot)] = (
                    self.parent.operator_set.bypass,
                    input_key
                )
            # add connections for outputs
            for output_key, output_slot in pipeline.outputs.items():
                self.parent.dsk[output_key] = (
                    self.parent.operator_set.bypass,
                    self.parent._mangle_output_slot(output_slot)
                )

    def __init__(self, stage: Stage, operator_set: DaskOperatorSet = TransformDaskOperatorSet()):
        self.stage = stage
        self.operator_set = operator_set
        self.dsk: MutableMapping[str, Any] = dict()
        self.block_ids: Dict[BaseBlock, int] = dict()
        self.visitor = self.Visitor(self)

    def _mangle_block(self, block: BaseBlock) -> str:
        return f'block_{self.block_ids[block]}'

    def _mangle_output_slot(self, output_slot) -> str:
        return self._mangle_block(output_slot.parent_block) + '_' + output_slot.meta.name + '_out'

    def _mangle_input_slot(self, input_slot) -> str:
        return self._mangle_block(input_slot.parent_block) + '_' + input_slot.meta.name + '_in'

    def __call__(self, pipeline: BasePipeline) -> Mapping[str, Any]:
        pipeline.accept(self.visitor)
        return self.dsk

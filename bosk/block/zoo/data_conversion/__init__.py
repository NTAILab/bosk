from typing import List, Optional, Tuple

import numpy as np

from ...base import BaseBlock, BlockInputData, TransformOutputData
from ...placeholder import PlaceholderMixin
from ...meta import BlockMeta, DynamicBlockMetaStub, make_simple_meta, BlockExecutionProperties
from ....data import BaseData, GPUData, CPUData, jnp


__all__ = [
    "Concat",
    "Argmax",
    "Average",
    "Stack",
    "Reshape",
    "Flatten",
    # for backward compatibility:
    "ConcatBlock",
    "ArgmaxBlock",
    "AverageBlock",
    "StackBlock",
    "ReshapeBlock",
    "FlattenBlock",
]


class Reshape(PlaceholderMixin, BaseBlock):
    """Reshaping block.

    Dynamically specifies the meta information.

    Args:
        new_shape: New data shape.
        input_name: Input slot name.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `input_name` (default="X"): Feature data.

    Output slots
    ------------

        - `input_name` (default="X"): Reshaped feature data.

    """
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, new_shape: Tuple[int], input_name: str = 'X'):
        self.meta = make_simple_meta([input_name], [input_name],
                                     execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))
        super().__init__()
        self.new_shape = new_shape

    def fit(self, _inputs: BlockInputData) -> 'ReshapeBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert len(inputs) == 1
        name, inp = next(iter(inputs.items()))
        reshaped: BaseData
        if isinstance(inp, CPUData):
            reshaped = CPUData(inp.data.reshape(self.new_shape))
        elif isinstance(inp, GPUData):
            reshaped = GPUData(inp.data.reshape(self.new_shape))
        else:
            raise NotImplementedError(f'Not implemented for input type: {type(inp)!r}')

        return {name: reshaped}


class Flatten(PlaceholderMixin, BaseBlock):
    """Flattening block. Flattens all dimensions except the first one.

    Dynamically specifies the meta information.

    Args:
        input_name: Input slot name.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `input_name` (default="X"): Feature data.

    Output slots
    ------------

        - `input_name` (default="X"): Flattened feature data.

    """
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, input_name: str = 'X'):
        self.meta = make_simple_meta([input_name], [input_name],
                                     execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'FlattenBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert len(inputs) == 1
        name, inp = next(iter(inputs.items()))
        reshaped: BaseData
        if isinstance(inp, CPUData):
            reshaped = CPUData(inp.data.reshape((inp.data.shape[0], -1)))
        elif isinstance(inp, GPUData):
            reshaped = GPUData(inp.data.reshape((inp.data.shape[0], -1)))
        else:
            raise NotImplementedError(f'Not implemented for input type: {type(inp)!r}')

        return {name: reshaped}


class Concat(PlaceholderMixin, BaseBlock):
    """Concatenation block.

    Dynamically specifies the meta information.

    Args:
        input_names: List of input slot names.
        axis: Axis along which to concatenate.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - `input_names[0]`: Data array 0.
        - ...
        - `input_names[n]`: Data array n.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `input_names[0]`: Data array 0.
        - ...
        - `input_names[n]`: Data array n.

    Output slots
    ------------

        - output: Concatenated data array.

    """
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'],
                                     execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))
        super().__init__()
        self.axis = axis
        self.ordered_input_names: Optional[List[str]] = None

    def fit(self, inputs: BlockInputData) -> 'ConcatBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        input_type = type(next(iter(inputs.values())))
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], input_type):
                ordered_inputs.append(inputs[name].data)
            else:
                raise ValueError(
                    "All inputs must be of the same type (CPUData or GPUData). "
                    f"Current input type: {type(inputs[name])!r}, "
                    f" expected: {input_type!r}"
                )
        concatenated: BaseData
        if input_type == CPUData:
            concatenated = CPUData(np.concatenate(ordered_inputs, axis=self.axis))
        elif input_type == GPUData:
            concatenated = GPUData(jnp.concatenate(ordered_inputs, axis=self.axis))
        else:
            raise ValueError(f"Unexpected input type: {input_type}")

        return {'output': concatenated}


class Stack(PlaceholderMixin, BaseBlock):
    """Stacking block.

    Dynamically specifies the meta information.

    Args:
        input_names: List of input slot names.
        axis: Axis along which to stack.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - `input_names[0]`: Data array 0.
        - ...
        - `input_names[n]`: Data array n.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - `input_names[0]`: Data array 0.
        - ...
        - `input_names[n]`: Data array n.

    Output slots
    ------------

        - output: Stacked data array.

    """
    meta: BlockMeta = DynamicBlockMetaStub()

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'],
                                     execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))
        super().__init__()
        self.axis = axis
        self.ordered_input_names: Optional[List[str]] = None

    def fit(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        input_type = type(next(iter(inputs.values())))
        ordered_inputs = []
        for name in self.ordered_input_names:
            if isinstance(inputs[name], input_type):
                ordered_inputs.append(inputs[name].data)
            else:
                raise ValueError("All inputs must be of the same type (CPUData or GPUData)")
        stacked: BaseData
        if input_type == CPUData:
            stacked = CPUData(np.stack(ordered_inputs, axis=self.axis))
        elif input_type == GPUData:
            stacked = GPUData(jnp.stack(ordered_inputs, axis=self.axis))
        else:
            raise ValueError(f"Unexpected input type: {input_type}")
        return {'output': stacked}


class Average(PlaceholderMixin, BaseBlock):
    """Averaging block.

    Args:
        axis: Axis along which to average.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Feature data array.

    Output slots
    ------------

        - output: Averaged feature data array.

    """
    meta = make_simple_meta(['X'], ['output'], execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'AverageBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        input_type = type(inputs['X'])
        if input_type not in [CPUData, GPUData]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        averaged = inputs['X'].data.mean(axis=self.axis)
        if input_type == CPUData:
            return {'output': CPUData(averaged)}
        else:
            return {'output': GPUData(averaged)}


class Argmax(PlaceholderMixin, BaseBlock):
    """Argmax block.

    Args:
        axis: Axis along which to calculate argmax.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        The fit step is bypassed.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Feature data array.

    Output slots
    ------------

        - output: Argmax result.

    """
    meta = make_simple_meta(['X'], ['output'], execution_props=BlockExecutionProperties(cpu=True, gpu=True, plain=True))

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'ArgmaxBlock':
        """The block bypasses the fit step."""
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        input_type = type(inputs['X'])
        if input_type not in [CPUData, GPUData]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        ids = inputs['X'].data.argmax(axis=self.axis)
        if input_type == CPUData:
            return {'output': CPUData(ids)}
        else:
            return {'output': GPUData(ids)}


ConcatBlock = Concat
ArgmaxBlock = Argmax
AverageBlock = Average
StackBlock = Stack
ReshapeBlock = Reshape
FlattenBlock = Flatten


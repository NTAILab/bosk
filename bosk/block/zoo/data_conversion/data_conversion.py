from typing import List

import numpy as np

from bosk.block import BaseBlock, BlockInputData, TransformOutputData
from bosk.block.meta import make_simple_meta
from bosk.data import CPUData


class ConcatBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'])
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit_gpu(self, inputs: BlockInputData) -> 'ConcatBlock':
        return self.fit(inputs)

    # def _gpu_concatenate(self, gpu_data_list, axis: int = -1):
    #     assert all(isinstance(gd, GPUData) for gd in gpu_data_list)
    #     gpu_context = gpu_data_list[0].context
    #     gpu_queue = gpu_data_list[0].queue
    #     if len(gpu_data_list) == 1:
    #         return gpu_data_list[0]
    #     shape = tuple(gpu_data_list[0].data.shape)
    #     total_axis = int(sum([gd.data.nbytes for gd in gpu_data_list]) / np.dtype(gpu_data_list[0].data.dtype).itemsize)
    #     assert total_axis / len(gpu_data_list) == np.prod(shape)
    #     new_shape = list(shape)
    #     new_shape[axis] = total_axis // np.prod(shape[:axis] + shape[axis + 1:])
    #
    #     # Создаем новый буфер
    #     gpu_data = GPUData(np.empty(shape=tuple(new_shape), dtype=gpu_data_list[0].data.dtype), gpu_context, gpu_queue)
    #     offset = 0
    #     kernel_code = """
    #         __kernel void concatenate(__global const float *src, int src_offset, __global float *dst, int dst_offset, int axis_len) {
    #             for (int i = 0; i < axis_len; i++) {
    #                 dst[dst_offset + i] = src[src_offset + i];
    #             }
    #         }
    #     """
    #     program = cl.Program(gpu_context, kernel_code).build()
    #     for gd in gpu_data_list:
    #         print("offset src ", offset * np.prod(shape[:axis + 1]))
    #         print("offset dst ", offset * np.prod(new_shape[:axis + 1]))
    #         print("axis len ", gd.data.shape[axis])
    #         print(gd.data.shape)
    #         print(gpu_data.data.shape)
    #
    #
    #         # Копируем данные из буферов gpu_data_list в новый буфер по нужной оси
    #         event = program.concatenate(gpu_queue, gd.data.shape[axis], None, gd.data.flatten(),
    #                                     offset * np.prod(shape[:axis + 1]), gpu_data.data.flatten(),
    #                                     offset * np.prod(new_shape[:axis + 1]), gd.data.shape[axis])
    #         event.wait()
    #         offset += gd.data.shape[axis]
    #     return gpu_data

    def transform_gpu(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        gpu_inputs = [inputs[name] for name in self.ordered_input_names]
        for gpu_data in gpu_inputs:
            if isinstance(gpu_data, CPUData):
                raise Exception("Error!")

        # gpu_data = np.concatenate(gpu_inputs, self.axis + 1)
        ordered_inputs = tuple(
            gpu_input.data
            for gpu_input in gpu_inputs
        )
        concatenated = np.concatenate(ordered_inputs, axis=self.axis)
        return {'output': concatenated}

    def fit(self, inputs: BlockInputData) -> 'ConcatBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = tuple(
            inputs[name].data
            for name in self.ordered_input_names
        )
        concatenated = np.concatenate(ordered_inputs, axis=self.axis)
        return {'output': concatenated}


class StackBlock(BaseBlock):
    meta = None

    def __init__(self, input_names: List[str], axis: int = -1):
        self.meta = make_simple_meta(input_names, ['output'])
        super().__init__()
        self.axis = axis
        self.ordered_input_names = None

    def fit(self, inputs: BlockInputData) -> 'StackBlock':
        self.ordered_input_names = list(inputs.keys())
        self.ordered_input_names.sort()
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert self.ordered_input_names is not None
        ordered_inputs = tuple(
            inputs[name]
            for name in self.ordered_input_names
        )
        stacked = np.stack(ordered_inputs, axis=self.axis)
        return {'output': stacked}


class AverageBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'AverageBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        averaged = inputs['X'].mean(axis=self.axis)
        return {'output': averaged}


class ArgmaxBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis

    def fit(self, _inputs: BlockInputData) -> 'ArgmaxBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        assert 'X' in inputs
        ids = inputs['X'].argmax(axis=self.axis)
        return {'output': ids}

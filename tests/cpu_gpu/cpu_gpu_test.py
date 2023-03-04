import numpy as np

from bosk.block.zoo.gpu_blocks import MoveToBlock
from bosk.block.zoo.input_plugs import InputBlock
from bosk.data import BaseData, CPUData, GPUData
from ..utility import log_test_name


def data_transfer_test():
    log_test_name()
    # Transfer data from CPU to GPU
    np.random.seed(42)
    data = np.random.normal(0, 100, 1000).astype(np.float32)
    base_data = BaseData(data)
    gpu_data = base_data.to_gpu()
    #
    # Transfer data back from GPU to CPU
    cpu_data = gpu_data.to_cpu()
    assert np.sum(cpu_data.data - data) == 0, 'The data was corrupted during the cpu->gpu->cpu transfer'

    input_data = BaseData(np.array([1, 2, 3]))
    input_block = InputBlock()
    move_to_block_cpu = MoveToBlock('CPU')
    move_to_block_gpu = MoveToBlock('GPU')

    inputs = {'X': input_data}
    inputs = input_block.transform(inputs)
    inputs = move_to_block_cpu.transform(inputs)
    assert isinstance(inputs['X'], CPUData), "MoveToBlock didn't move the data to the cpu"

    inputs = {'X': input_data}
    inputs = input_block.transform(inputs)
    inputs = move_to_block_gpu.transform(inputs)
    assert isinstance(inputs['X'], GPUData), "MoveToBlock didn't move the data to the gpu"

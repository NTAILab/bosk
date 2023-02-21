import numpy as np

from bosk.block.zoo.gpu_blocks import MoveToBlock
from bosk.block.zoo.input_plugs import InputBlock
from bosk.data import BaseData, CPUData, GPUData


def data_transfer_test():
    # Transfer data from CPU to GPU
    np.random.seed(42)
    data = np.random.normal(0, 100, 1000).astype(np.float32)
    base_data = BaseData(data)
    gpu_data = base_data.to_gpu()

    # Transfer data back from GPU to CPU
    cpu_data = gpu_data.to_cpu()
    assert np.sum(cpu_data.data - data) == 0, 'The data was corrupted during the cpu->gpu->cpu transfer'

    # Transfer data from GPU to GPU with the same context and queue
    gpu_data2 = base_data.to_gpu(context=gpu_data.context, queue=gpu_data.queue)
    assert gpu_data.context == gpu_data2.context, 'The context was corrupted during the gpu->gpu transfer'
    assert gpu_data.queue == gpu_data2.queue, 'The queue was corrupted during the gpu->gpu transfer'

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

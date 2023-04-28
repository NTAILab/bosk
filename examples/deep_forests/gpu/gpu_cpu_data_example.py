import numpy as np

from bosk.block.zoo.gpu_blocks import MoveToBlock
from bosk.block.zoo.input_plugs import InputBlock
from bosk.data import BaseData, CPUData, GPUData

if __name__ == "__main__":
    # Example usage:
    # Transfer data from CPU to GPU
    data = np.array([1, 2, 3, 4], dtype=np.float32)
    base_data = BaseData(data)
    gpu_data = base_data.to_gpu()

    # Transfer data back from GPU to CPU
    cpu_data = gpu_data.to_cpu()

    # Transfer data from GPU to GPU with the same context and queue
    gpu_data2 = base_data.to_gpu()

    input_data = BaseData(np.array([1, 2, 3]))
    input_block = InputBlock()
    move_to_block_cpu = MoveToBlock('CPU')
    move_to_block_gpu = MoveToBlock('GPU')

    inputs = {'X': input_data}
    inputs = input_block.transform(inputs)
    inputs = move_to_block_cpu.transform(inputs)
    assert isinstance(inputs['X'], CPUData)

    inputs = {'X': input_data}
    inputs = input_block.transform(inputs)
    inputs = move_to_block_gpu.transform(inputs)
    assert isinstance(inputs['X'], GPUData)

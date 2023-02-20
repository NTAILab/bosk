from typing import Any, Optional

import pyopencl as cl
import pyopencl.array as cla
import numpy as np

Data = Any
"""Type for data which will be transferred between blocks.
"""


class BaseData:
    def __init__(self, data: np.ndarray):
        self.data = data

    def to_cpu(self) -> 'CPUData':
        """Transfers data to a CPU-based representation."""
        return CPUData(self.data)

    def to_gpu(self, context: Optional[cl.Context] = None, queue: Optional[cl.CommandQueue] = None) -> 'GPUData':
        """Transfers data to a GPU-based representation.
        If context and queue are not provided, a new context and queue will be created.
        """
        context = context or cl.create_some_context()
        queue = queue or cl.CommandQueue(context)
        return GPUData(self.data, context, queue)


class CPUData(BaseData):
    def __init__(self, data: np.ndarray):
        super().__init__(data)


class GPUData(BaseData):
    def __init__(self, data: Any, context: Optional[cl.Context] = None,
                 queue: Optional[cl.CommandQueue] = None):
        super().__init__(data)
        context = context or cl.create_some_context()
        queue = queue or cl.CommandQueue(context)
        self.context = context
        self.queue = queue
        if isinstance(data, cla.Array):
            data = data.get()
        self.data = cla.to_device(queue, data)

    def to_cpu(self) -> 'CPUData':
        """Transfers data to a CPU-based representation."""
        return CPUData(self.data.get())

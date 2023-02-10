from typing import Any, Optional

import pyopencl as cl
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
        buf = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.data)
        return GPUData(self.data, context, queue, buf)


class CPUData(BaseData):
    def __init__(self, data: np.ndarray):
        super().__init__(data)


class GPUData(BaseData):
    def __init__(self, data: np.ndarray, context: cl.Context, queue: cl.CommandQueue, buf: cl.Buffer):
        super().__init__(data)
        self.context = context
        self.queue = queue
        self.buf = buf

    def to_cpu(self) -> 'CPUData':
        """Transfers data to a CPU-based representation."""
        data = np.empty_like(self.data)
        cl.enqueue_copy(self.queue, data, self.buf)
        return CPUData(data)

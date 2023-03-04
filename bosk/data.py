from typing import Any, Union

import jax.numpy as jnp

import numpy as np

Data = Any
"""Type for data which will be transferred between blocks.
"""


class BaseData:
    def __init__(self, data: Union[np.ndarray, jnp.ndarray]):
        # if not isinstance(data, (np.ndarray, jnp.ndarray)):
        #     raise TypeError(f"Data type {type(data)} not supported.")
        self.data = data

    def to_cpu(self) -> 'CPUData':
        """Returns self, since the data is already on CPU."""
        return CPUData(self.data)

    def to_gpu(self) -> 'GPUData':
        """Transfers data to a GPU-based representation."""
        return GPUData(self.data)


class CPUData(BaseData):
    def __init__(self, data: Any):
        if isinstance(data, jnp.ndarray):
            data = np.array(data)
        super().__init__(data)


class GPUData(BaseData):
    def __init__(self, data: Any):
        if isinstance(data, np.ndarray):
            data = jnp.array(data)
        super().__init__(data)

    def to_cpu(self) -> 'CPUData':
        """Transfers data to a CPU-based representation."""
        return CPUData(np.array(self.data))

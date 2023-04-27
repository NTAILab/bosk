from __future__ import annotations
from typing import Optional

import jax.numpy as jnp
import numpy as np

from ...auto import auto_block
from ...meta import BlockExecutionProperties
from ....data import BaseData, CPUData, GPUData


@auto_block(auto_state=True,
            execution_props=BlockExecutionProperties(cpu=True, gpu=True),
            random_state_field=None)
class WeightsBlock:
    _weights: Optional[BaseData]

    def __init__(self, ord: int = 1, device="CPU"):
        self._weights = None
        self.ord = ord
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightsBlock':
        if self.device not in ["CPU", "GPU"]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        if self.device == "CPU":
            cpu_weights = 1 - (np.take_along_axis(X, y[:, np.newaxis], axis=1)) ** self.ord
            self._weights = CPUData(cpu_weights.reshape((-1,)))
        else:
            gpu_weights = 1 - (jnp.take_along_axis(X, y[:, jnp.newaxis], axis=1)) ** self.ord
            self._weights = GPUData(gpu_weights.reshape((-1,)))
        return self

    def transform(self, _X=None) -> BaseData:
        assert self._weights is not None
        return self._weights

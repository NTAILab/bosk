from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from bosk.block import auto_block
from bosk.block.meta import BlockExecutionProperties
from bosk.data import CPUData, GPUData


@auto_block(auto_state=True,
            execution_props=BlockExecutionProperties(cpu=True, gpu=True),
            random_state_field=None)
class WeightsBlock:
    def __init__(self, ord: int = 1, device="CPU"):
        self._weights = None
        self.ord = ord
        self.device = device

    def fit(self, X, y) -> 'WeightsBlock':
        if self.device not in ["CPU", "GPU"]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        if self.device == "CPU":
            weights = 1 - (np.take_along_axis(X, y[:, np.newaxis], axis=1)) ** self.ord
            self._weights = CPUData(weights.reshape((-1,)))
        else:
            weights = 1 - (jnp.take_along_axis(X, y[:, jnp.newaxis], axis=1)) ** self.ord
            self._weights = GPUData(weights.reshape((-1,)))
        return self

    def transform(self, _X = None) -> CPUData | GPUData:
        return self._weights

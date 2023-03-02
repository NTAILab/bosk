import jax.numpy as jnp
import numpy as np

from bosk.block import auto_block
from bosk.block.meta import BlockExecutionProperties
from bosk.data import CPUData, GPUData


@auto_block(execution_props=BlockExecutionProperties(cpu=True, gpu=True))
class WeightsBlock:
    def __init__(self, ord: int = 1):
        self._weights = None
        self.ord = ord

    def fit(self, X, y) -> 'WeightsBlock':
        input_type = type(X)
        if input_type not in [CPUData, GPUData]:
            raise TypeError("All inputs must be of type: CPUData or GPUData.")
        X_data = X.data
        y_data = y.data
        if input_type == CPUData:
            weights = 1 - (np.take_along_axis(X_data, y_data[:, np.newaxis], axis=1)) ** self.ord
            self._weights = CPUData(weights.reshape((-1,)))
        else:
            weights = 1 - (jnp.take_along_axis(X_data, y_data[:, jnp.newaxis], axis=1)) ** self.ord
            self._weights = CPUData(weights.reshape((-1,)))
        return self

    def transform(self, _X) -> CPUData | GPUData:
        return self._weights

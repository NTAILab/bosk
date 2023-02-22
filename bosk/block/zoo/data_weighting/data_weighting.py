import jax.numpy as jnp
import numpy as np

from bosk.block import auto_block


@auto_block
class WeightsBlock:
    def __init__(self, ord: int = 1):
        self._weights = None
        self.ord = ord

    def fit_transform(self, X, y):
        weights = 1 - (jnp.take_along_axis(X, y[:, jnp.newaxis], axis=1)) ** self.ord
        self._weights = weights.reshape((-1,))
        return self

    def transform_gpu(self, _X) -> 'jnp.array':
        return self._weights

    def fit(self, X, y) -> 'WeightsBlock':
        weights = 1 - (np.take_along_axis(X, y[:, np.newaxis], axis=1)) ** self.ord
        self._weights = weights.reshape((-1,))
        return self

    def transform(self, _X) -> 'np.ndarray':
        return self._weights

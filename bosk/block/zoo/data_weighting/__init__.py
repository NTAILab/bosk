from typing import Optional

import numpy as np

from ...auto import auto_block
from ...meta import BlockExecutionProperties
from ....data import BaseData, CPUData, GPUData, jnp


__all__ = ['WeightsBlock']


@auto_block(auto_state=True,
            execution_props=BlockExecutionProperties(cpu=True, gpu=True),
            random_state_field=None)
class WeightsBlock:
    """Data weighting block.

    Args:
        ord: Order of weighting function.
        device: Device for weights storage.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Feature data.
        - y: Label data.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - _X: Ignored.

    Output slots
    ------------

        - output: Sample weights.

    """
    _weights: Optional[BaseData]

    def __init__(self, ord: int = 1, device="CPU"):
        self._weights = None
        self.ord = ord
        self.device = device

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightsBlock':
        """Fit the weights block.

        Args:
            X: Feature data.
            y: Label data.

        Returns:
            Self.

        """
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
        """Transform: get the weights.

        Returns:
            Weights.

        """
        assert self._weights is not None
        return self._weights

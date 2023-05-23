from typing import Any, Union
import numpy as np
import warnings


try:
    import jax.numpy as jnp
except ModuleNotFoundError:
    warnings.warn("Cannot import jax library, please install it or don't use GPU blocks")
    jnp = np  # type: ignore



class BaseData:
    """Base class for data which will be transferred between blocks.

    Attributes:
        data: Underlying data.

    """
    def __init__(self, data: Union[np.ndarray, jnp.ndarray]):
        self.data = data

    def to_cpu(self) -> 'CPUData':
        """Convert data to CPU representation.

        Returns:
            CPU Data.

        """
        return CPUData(self.data)

    def to_gpu(self) -> 'GPUData':
        """Converts data to a GPU-based representation.

        Returns:
            GPU Data.

        """
        return GPUData(self.data)

    def __repr__(self) -> str:
        """Representation of data shape and type.

        Returns:
            String data representation.

        """
        return f'<{self.__class__.__name__} {self.data.shape!r} {self.data.dtype!r}>'


class CPUData(BaseData):
    """CPU-based representation of data.
    """

    data: np.ndarray

    def __init__(self, data: Union[np.ndarray, jnp.ndarray]):
        if isinstance(data, jnp.ndarray):
            data = np.array(data)
        super().__init__(data)


class GPUData(BaseData):
    """GPU-based (JAX) representation of data.
    """

    data: jnp.ndarray

    def __init__(self, data: Union[np.ndarray, jnp.ndarray]):
        if isinstance(data, np.ndarray):
            data = jnp.array(data)
        super().__init__(data)

    def to_cpu(self) -> 'CPUData':
        """Transfers data to a CPU-based representation."""
        return CPUData(np.array(self.data))

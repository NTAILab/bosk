from typing import Any


Data = Any
"""Type for data which will be transferred between blocks.
"""


class BaseData:
    ...  # common parameters


class CPUData(BaseData):
    ...  # cpu-specific parameters


class GPUData(BaseData):
    ...  # cpu-specific parameters

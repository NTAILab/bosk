from typing import Any

import torch

Data = Any
"""Type for data which will be transferred between blocks.
"""


class BaseData:
    def __init__(self, data: Data):
        self.data = data
        self.device = "cpu"

    def to(self, device):
        if hasattr(self.data, 'to'):
            self.data = self.data.to(device)
        else:
            self.data = torch.tensor(self.data).to(device)
        self.device = device


class CPUData(BaseData):
    def __init__(self, data: Data):
        super().__init__(data)
        self.to("cpu")


class GPUData(BaseData):
    def __init__(self, data: Data):
        super().__init__(data)
        self.to("cuda")


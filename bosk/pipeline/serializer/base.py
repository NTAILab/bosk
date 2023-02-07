from abc import ABC, abstractmethod
from ..base import BasePipeline


class BaseSerializer(ABC):
    @abstractmethod
    def dump(self, pipeline: BasePipeline, out_file):
        ...

    @abstractmethod
    def load(self, in_file) -> BasePipeline:
        ...

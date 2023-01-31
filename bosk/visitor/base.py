from abc import ABC, abstractmethod
from typing import Any


class BaseVisitor(ABC):
    """Base Visitor interface.

    Visitor can be used to process any entity of a computational pipeline,
    such as: block, connection, or even the whole pipeline.

    It is important to implement dispatching for all entity types or
    to make some default method which will not raise ``NotImplementedError``
    for unknown entity types.

    """
    @abstractmethod
    def visit(self, obj: Any):
        ...

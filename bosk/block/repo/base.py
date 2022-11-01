from ..base import BaseBlock
from abc import ABC, abstractmethod
from typing import Type



class BaseBlockClassRepository(ABC):
    """Base block class repository, the parent of every block class repo.

    It can be used to get block classes by name,
    for example in pipeline builders.

    """
    @abstractmethod
    def get(self, name: str) -> Type[BaseBlock]:
        """Get block class by name.

        Args:
            name: Block name.

        Returns:
            Demanded block class.

        """
        ...

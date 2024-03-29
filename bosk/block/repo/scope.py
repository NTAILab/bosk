from ..base import BaseBlock
from .base import BaseBlockClassRepository
from typing import Type


class ScopeBlockClassRepository(BaseBlockClassRepository):
    """Scope block class repository.

    Helps to get access to all blocks in a scope.

    Args:
        scope (dict): Scope dictionary.
                      Pass `locals()` or `globals()` to use the caller's scope.

    """
    def __init__(self, scope: dict):
        self.__scope = scope

    def get(self, name: str) -> Type[BaseBlock]:
        block_name = name + 'Block'
        block_cls = self.__scope.get(block_name, None)
        if block_cls is None:
            raise ValueError(
                f'Wrong block class: {name} ({block_name} not found). '
                'Probably name ends with "Block", in this case just remove this postfix.'
            )
        return block_cls

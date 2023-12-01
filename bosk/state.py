"""State of the package.
"""
from functools import lru_cache
from collections import deque
from typing import Optional, Any


class ActiveBuildersState:
    """Active builders state, used for builders context managers.
    """
    def __init__(self):
        self.__builders = deque()

    def peek(self) -> Optional[Any]:
        if len(self.__builders) == 0:
            return None
        return self.__builders[-1]

    def push(self, value):
        self.__builders.append(value)

    def pop(self):
        self.__builders.pop()


@lru_cache(maxsize=1)
class BoskState:
    """Bosk state singleton.
    """
    active_builders = ActiveBuildersState()


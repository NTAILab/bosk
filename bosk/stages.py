from dataclasses import dataclass
from enum import Enum, auto


class Stage(Enum):
    FIT = auto()
    TRANSFORM = auto()


@dataclass(eq=True, frozen=True)
class Stages:
    fit: bool = True
    transform: bool = True
    transform_on_fit: bool = False

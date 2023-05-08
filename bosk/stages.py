"""Pipeline execution stages.

"""
from dataclasses import dataclass
from enum import Enum, auto


class Stage(Enum):
    """Stage of pipeline execution: `FIT` or `TRANSFORM`.

    At the `FIT` stage the blocks are fitted to the data given by preceding blocks
    and then applied to transform the given data (predict).

    At the `TRANSFORM` stage the blocks are only used to transform
    the given by preceding blocks data.

    """
    FIT = auto()
    TRANSFORM = auto()


@dataclass(eq=True, frozen=True)
class Stages:
    """At which stages of pipeline execution slot value is required
    and should be passed to the block.

    Attributes:
        fit: Slot value is needed for the fit method.
        transform: Slot value is needed for the transform method.
        transform_on_fit: Slot value is needed for the transform method only at the `FIT` stage.
    """
    fit: bool = True
    transform: bool = True
    transform_on_fit: bool = False

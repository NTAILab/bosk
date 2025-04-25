"""Data routing blocks.

Can be used to determine and pass subsets of data to blocks.

For example, to implement stacing where each block is trained independently
on some subset of the data.

"""
from .cs import CS, CSJoin, CSFilter, CSBlock, CSJoinBlock, CSFilterBlock
from .cv import CVTrainIndices, SubsetTrainWrapper, CVTrainIndicesBlock, SubsetTrainWrapperBlock
from .shared import Shared


__all__ = [
    "CS",
    "CSJoin",
    "CSFilter",
    "CVTrainIndices",
    "SubsetTrainWrapper",
    "Shared",
    # for backward compatibility:
    "CSBlock",
    "CSJoinBlock",
    "CSFilterBlock",
    "CVTrainIndicesBlock",
    "SubsetTrainWrapperBlock",
]

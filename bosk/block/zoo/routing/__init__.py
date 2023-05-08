"""Data routing blocks.

Can be used to determine and pass subsets of data to blocks.

For example, to implement stacing where each block is trained independently
on some subset of the data.

"""
from .cs import CSBlock, CSJoinBlock, CSFilterBlock
from .cv import CVTrainIndicesBlock, SubsetTrainWrapperBlock


__all__ = [
    "CSBlock",
    "CSJoinBlock",
    "CSFilterBlock",
    "CVTrainIndicesBlock",
    "SubsetTrainWrapperBlock",
]

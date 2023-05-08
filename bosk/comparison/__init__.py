"""Deep Forest based model comparison module.

It is useful for performance and quality evaluation and comparison of Deep Forest models
between each other and with third-party models.

Besides cross-validated evaluation, it can find and extract common part of different Deep Forest models
(defined with `bosk`) and precompute its result for each train-test split.
It allows to significantly speed up model comparison.

"""

from .base import BaseForeignModel, BaseComparator
from .cross_val import CVComparator
from .metric import BaseMetric, MetricWrapper

__all__ = [
    "BaseForeignModel",
    "BaseComparator",
    "CVComparator",
    "BaseMetric",
    "MetricWrapper",
]

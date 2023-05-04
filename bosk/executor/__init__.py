"""Pipeline execution module.

Any executor can be initialized with a pipeline and a stage.
Optionally, input and output pipeline slots lists can be specified.

An initialized pipeline executor acts like a function and can be applied
to a dictionary of input values.

The output of the executor is a special dictionary of output values,
which contain wrapped data (:py:class:`bosk.data.BaseData`).
In order to obtain NumPy arrays as output, the `.numpy()` method should be called
on the result.

Example of usage:

.. code-block:: python

    pipeline = make_pipeline()  # make a pipeline somehow
    fitter = TopologicalExecutor(pipeline, stage=Stage.FIT)
    fitter({'X': X_train, 'y': y_train})  # fit on dictionary of input numpy arrays
    predictor = TopologicalExecutor(pipeline, stage=Stage.TRANSFORM)
    predictions = predictor({'X': X_test}).numpy()  # result: dictionary of output numpy arrays

"""

from .base import BaseSlotHandler, DefaultSlotHandler
from .block import BaseBlockExecutor, DefaultBlockExecutor, FitBlacklistBlockExecutor
from .timer import TimerBlockExecutor
from .sklearn_interface import BaseBoskPipelineWrapper, BoskPipelineClassifier, BoskPipelineRegressor, BoskPipelineTransformer
from .recursive import RecursiveExecutor
from .topological import TopologicalExecutor
from .parallel import GreedyParallelExecutor

__all__ = [
    "parallel",
    "BaseSlotHandler",
    "DefaultSlotHandler",
    "BaseBlockExecutor",
    "DefaultBlockExecutor",
    "FitBlacklistBlockExecutor",
    "TimerBlockExecutor",
    "BaseBoskPipelineWrapper",
    "BoskPipelineClassifier",
    "BoskPipelineRegressor",
    "BoskPipelineTransformer",
    "RecursiveExecutor",
    "TopologicalExecutor",
    "GreedyParallelExecutor"
]

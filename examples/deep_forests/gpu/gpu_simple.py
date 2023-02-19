"""Example of the basic deep forest creating using either manual graph definition and the functional API.
"""

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.base import BaseExecutor
from bosk.stages import Stage
from bosk.executor.descriptor import HandlingDescriptor
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder

from bosk.block.zoo.gpu_blocks.transition import MoveToBlock
from bosk.data import CPUData, GPUData


def make_deep_forest_functional_cpu(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    X = b.MoveTo("CPU")(X=X)
    # rf_1 = b.RFC(random_state=42)(X=X, y=y)
    # et_1 = b.ETC(random_state=42)(X=X, y=y)
    concat_1 = b.Concat(['X', 'X_1', 'X_2'])(X=X, X_1=X, X_2=X)

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'concat': concat_1}
        ),
        HandlingDescriptor.from_classes(Stage.FIT),
        inputs=['X', 'y'],
        outputs=['concat'],
        **ex_kw
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'concat': concat_1}
        ),
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs=['X'],
        outputs=['concat'],
        **ex_kw
    )
    return fit_executor, transform_executor


def make_deep_forest_functional_gpu(executor, **ex_kw):
    b = FunctionalPipelineBuilder()
    X, y = b.Input()(), b.TargetInput()()
    X = b.MoveTo("GPU")(X=X)
    # rf_1 = b.RFC(random_state=42)(X=X, y=y)
    # et_1 = b.ETC(random_state=42)(X=X, y=y)
    concat_1 = b.Concat(['X', 'X_1', 'X_2'])(X=X, X_1=X, X_2=X)

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'concat': concat_1}
        ),
        HandlingDescriptor.from_classes(Stage.FIT),
        inputs=['X', 'y'],
        outputs=['concat'],
        **ex_kw
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'concat': concat_1}
        ),
        HandlingDescriptor.from_classes(Stage.TRANSFORM),
        inputs=['X'],
        outputs=['concat'],
        **ex_kw
    )
    return fit_executor, transform_executor


def run_cpu():
    executor_class = RecursiveExecutor
    fit_executor, transform_executor = make_deep_forest_functional_cpu(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': CPUData(train_X), 'y': CPUData(train_y)})
    print("Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    test_result = transform_executor({'X': CPUData(test_X)})


def run_gpu():
    executor_class = RecursiveExecutor
    fit_executor, transform_executor = make_deep_forest_functional_gpu(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': GPUData(train_X), 'y': GPUData(train_y)})
    print("Fit successful")
    train_result = transform_executor({'X': GPUData(train_X)})
    print(train_result["concat"].data)
    test_result = transform_executor({'X': GPUData(test_X)})


if __name__ == "__main__":
    import pyopencl as cl

    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            print(device.name)

    # run_cpu()
    run_gpu()

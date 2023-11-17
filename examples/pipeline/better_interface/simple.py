"""Example of deep forest pipeline construction with "better" functional interface.

"""
import json
from bosk.data import CPUData
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bosk.executor.recursive import RecursiveExecutor

from sklearn.metrics import roc_auc_score
from bosk.block import BaseBlock
from functools import wraps
from typing import Mapping

from bosk.pipeline.builder.functional import FunctionalPipelineBuilder
from bosk.pipeline.builder.eager import EagerPipelineBuilder
from bosk.data import BaseData
from bosk.pipeline.connection import Connection
from bosk.executor.block import BaseBlockExecutor, InputSlotToDataMapping
from bosk.executor.block import DefaultBlockExecutor
from bosk.block.functional import FunctionalBlockWrapper
from bosk.block.eager import EagerBlockWrapper
from bosk.block.base import BaseInputBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import AverageBlock, ConcatBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.metrics import RocAucBlock


global_active_builder = None


def get_active_builder():
    global global_active_builder
    return global_active_builder


def set_active_builder(value):
    global global_active_builder
    global_active_builder = value



class ContextManagerBuilderMixin:
    def __enter__(self):
        if get_active_builder() is not None:
            raise Exception(
                f'Nested pipeline builders with context manager interface are not allowed'
            )
        set_active_builder(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert get_active_builder() is not None
        set_active_builder(None)
        return False


class BetterFunctionalPipelineBuilder(ContextManagerBuilderMixin, FunctionalPipelineBuilder):
    ...


class BetterEagerPipelineBuilder(ContextManagerBuilderMixin, EagerPipelineBuilder):
    ...


class PlaceholderFunction:
    def __init__(self, block):
        self.block = block
        self._get_builder()._register_block(self.block)

    def _get_builder(self):
        builder = get_active_builder()
        assert builder is not None, \
            'No active builder found. ' \
            'Please, enter a builder scope by `with BetterFunctionalPipelineBuilder() as b:`'
        return builder

    def __call__(self, *pfn_args, **pfn_kwargs):
        builder = self._get_builder()
        if len(pfn_args) > 0:
            assert len(pfn_kwargs) == 0, \
                'Either unnamed or named arguments can be used, but not at the same time'
            assert len(pfn_args) == 1, \
                'Only one unnamed argument is supported (we can infer name only in this case)'
            assert isinstance(self.block, BaseInputBlock)
            pfn_kwargs = {
                self.block.get_single_input().meta.name: pfn_args[0]
            }
        need_construct_eager = False
        block_input_mapping: InputSlotToDataMapping = dict()
        for input_name, input_block_wrap_or_data in pfn_kwargs.items():
            block_input = self.block.slots.inputs[input_name]
            is_functional = isinstance(input_block_wrap_or_data, FunctionalBlockWrapper)
            is_eager = isinstance(input_block_wrap_or_data, EagerBlockWrapper)
            if is_functional or is_eager:
                builder._connections.append(
                    Connection(
                        src=input_block_wrap_or_data.get_output_slot(),
                        dst=block_input,
                    )
                )
                if is_eager:
                    block_input_mapping[block_input] = input_block_wrap_or_data.get_output_data()
            elif isinstance(input_block_wrap_or_data, BaseData):
                block_input_mapping[block_input] = input_block_wrap_or_data
                is_eager = True
            else:
                raise ValueError(
                    f'Wrong placeholder input type: {type(input_block_wrap_or_data)}'
                )
            need_construct_eager |= is_eager
        if need_construct_eager:
            assert isinstance(builder, EagerPipelineBuilder), \
                'Only eager pipeline builder can process Eager block wrappers'
            block_wrapper = EagerBlockWrapper(self.block, executor=builder.block_executor)
            if len(block_input_mapping) > 0:
                block_wrapper.execute(block_input_mapping)
        else:
            block_wrapper = FunctionalBlockWrapper(self.block)
        return block_wrapper


class Input(PlaceholderFunction):
    @wraps(InputBlock.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(InputBlock(*args, **kwargs))


class TargetInput(PlaceholderFunction):
    @wraps(TargetInputBlock.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(TargetInputBlock(*args, **kwargs))
        

class RFC(PlaceholderFunction):
    @wraps(RFCBlock.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(RFCBlock(*args, **kwargs))


class ETC(PlaceholderFunction):
    @wraps(ETCBlock.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(ETCBlock(*args, **kwargs))
        

class Concat(PlaceholderFunction):
    @wraps(ConcatBlock.__init__)
    def __init__(self, *args, **kwargs):
        self.block = ConcatBlock(*args, **kwargs)


class Stack(PlaceholderFunction):
    @wraps(StackBlock.__init__)
    def __init__(self, *args, **kwargs):
        self.block = StackBlock(*args, **kwargs)


class Average(PlaceholderFunction):
    @wraps(AverageBlock.__init__)
    def __init__(self, *args, **kwargs):
        self.block = AverageBlock(*args, **kwargs)


class Argmax(PlaceholderFunction):
    @wraps(ArgmaxBlock.__init__)
    def __init__(self, *args, **kwargs):
        self.block = ArgmaxBlock(*args, **kwargs)


class RocAuc(PlaceholderFunction):
    @wraps(RocAucBlock.__init__)
    def __init__(self, *args, **kwargs):
        self.block = RocAucBlock(*args, **kwargs)


def make_deep_forest_functional(executor, forest_params=None, **ex_kw):
    if forest_params is None:
        forest_params = dict()

    with BetterFunctionalPipelineBuilder() as b:
        X, y = Input()(), TargetInput()()
        rf_1 = RFC(random_state=42, **forest_params)(X=X, y=y)
        et_1 = ETC(random_state=42, **forest_params)(X=X, y=y)
        concat_1 = Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
        rf_2 = RFC(random_state=42, **forest_params)(X=concat_1, y=y)
        et_2 = ETC(random_state=42, **forest_params)(X=concat_1, y=y)
        concat_2 = Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
        rf_3 = RFC(random_state=42, **forest_params)(X=concat_2, y=y)
        et_3 = ETC(random_state=42, **forest_params)(X=concat_2, y=y)
        stack_3 = Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)

        average_3 = Average(axis=1)(X=stack_3)
        argmax_3 = Argmax(axis=1)(X=average_3)

        rf_1_roc_auc = RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        stage=Stage.FIT,
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
        **ex_kw
    )
    transform_executor = executor(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        stage=Stage.TRANSFORM,
        inputs=['X'],
        outputs=['probas', 'labels'],
        **ex_kw
    )
    return fit_executor, transform_executor


def main_functional():
    fit_executor, transform_executor = make_deep_forest_functional(RecursiveExecutor)

    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    fit_result = fit_executor({'X': CPUData(train_X), 'y': CPUData(train_y)})
    print("  Fit successful")
    train_result = transform_executor({'X': CPUData(train_X)})
    print(
        "  Fit probas == probas on train:",
        np.allclose(fit_result['probas'].data, train_result['probas'].data)
    )
    test_result = transform_executor({'X': CPUData(test_X)})
    print("  Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    print("  Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


def main_eager():
    # prepare data first
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    executor_cls = RecursiveExecutor
    block_executor = DefaultBlockExecutor()
    forest_params = dict()

    with BetterEagerPipelineBuilder(block_executor) as b:
        X, y = Input()(CPUData(train_X)), TargetInput()(CPUData(train_y))
        # alternative ways to define input blocks:
        # X, y = Input()(X=CPUData(train_X)), TargetInput()(y=CPUData(train_y))
        # or without input data:
        # X, y = Input()(), TargetInput()()
        # X.execute({X.block.get_single_input(): CPUData(train_X)})
        # y.execute({y.block.get_single_input(): CPUData(train_y)})

        rf_1 = RFC(random_state=42, **forest_params)(X=X, y=y)
        et_1 = ETC(random_state=42, **forest_params)(X=X, y=y)
        concat_1 = Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
        rf_2 = RFC(random_state=42, **forest_params)(X=concat_1, y=y)
        et_2 = ETC(random_state=42, **forest_params)(X=concat_1, y=y)
        concat_2 = Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
        rf_3 = RFC(random_state=42, **forest_params)(X=concat_2, y=y)
        et_3 = ETC(random_state=42, **forest_params)(X=concat_2, y=y)
        stack_3 = Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
        average_3 = Average(axis=1)(X=stack_3)
        argmax_3 = Argmax(axis=1)(X=average_3)

        rf_1_roc_auc = RocAuc()(gt_y=y, pred_probas=rf_1)
        roc_auc = RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = executor_cls(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'rf_1_roc-auc': rf_1_roc_auc, 'roc-auc': roc_auc}
        ),
        stage=Stage.FIT,
        inputs=['X', 'y'],
        outputs=['probas', 'rf_1_roc-auc', 'roc-auc'],
    )
    transform_executor = executor_cls(
        b.build(
            {'X': X, 'y': y},
            {'probas': average_3, 'labels': argmax_3}
        ),
        stage=Stage.TRANSFORM,
        inputs=['X'],
        outputs=['probas', 'labels'],
    )

    train_result = transform_executor({'X': CPUData(train_X)})
    print(
        "  Fit probas == probas on train:",
        np.allclose(average_3.get_output_data().data, train_result['probas'].data)
    )
    test_result = transform_executor({'X': CPUData(test_X)})
    print("  Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'].data[:, 1]))
    print("  Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'].data[:, 1]))


if __name__ == "__main__":
    main_functional()
    main_eager()


"""Example of simple Confidence Screening Deep Forest definition.

"""
from typing import Callable, Optional

import numpy as np

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.block import BaseBlock
from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.naive import NaiveExecutor
from bosk.stages import Stage
from bosk.slot import BlockOutputSlot
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock, AccuracyBlock, F1ScoreBlock
from bosk.block.zoo.routing import CSBlock, CSJoinBlock, CSFilterBlock
from bosk.block.zoo.multi_grained_scanning.multi_grained_scanning import MultiGrainedScanningBlock


def make_deep_forest():
    input_x = InputBlock()
    input_y = TargetInputBlock()
    rf_1 = RFCBlock(seed=42)
    et_1 = ETCBlock(seed=42)
    concat_1 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_2 = RFCBlock(seed=42)
    et_2 = ETCBlock(seed=42)
    concat_2 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_3 = RFCBlock(seed=42)
    et_3 = ETCBlock(seed=42)
    stack_3 = StackBlock(['X_0', 'X_1'], axis=1)
    average_3 = AverageBlock(axis=1)
    argmax_3 = ArgmaxBlock(axis=1)
    pipeline = BasePipeline(
        nodes=[
            input_x,
            input_y,
            rf_1,
            et_1,
            concat_1,
            rf_2,
            et_2,
            concat_2,
            rf_3,
            et_3,
            stack_3,
            average_3,
            argmax_3,
        ],
        connections=[
            # input X
            Connection(input_x.meta.outputs['X'], rf_1.meta.inputs['X']),
            Connection(input_x.meta.outputs['X'], et_1.meta.inputs['X']),
            # input y
            Connection(input_y.meta.outputs['y'], rf_1.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], et_1.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], rf_2.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], et_2.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], rf_3.meta.inputs['y']),
            Connection(input_y.meta.outputs['y'], et_3.meta.inputs['y']),
            # layers connection
            Connection(rf_1.meta.outputs['output'], concat_1.meta.inputs['X_0']),
            Connection(et_1.meta.outputs['output'], concat_1.meta.inputs['X_1']),
            Connection(concat_1.meta.outputs['output'], rf_2.meta.inputs['X']),
            Connection(concat_1.meta.outputs['output'], et_2.meta.inputs['X']),
            Connection(rf_2.meta.outputs['output'], concat_2.meta.inputs['X_0']),
            Connection(et_2.meta.outputs['output'], concat_2.meta.inputs['X_1']),
            Connection(concat_2.meta.outputs['output'], rf_3.meta.inputs['X']),
            Connection(concat_2.meta.outputs['output'], et_3.meta.inputs['X']),
            Connection(rf_3.meta.outputs['output'], stack_3.meta.inputs['X_0']),
            Connection(et_3.meta.outputs['output'], stack_3.meta.inputs['X_1']),
            Connection(stack_3.meta.outputs['output'], average_3.meta.inputs['X']),
            Connection(average_3.meta.outputs['output'], argmax_3.meta.inputs['X']),
        ]
    )

    fit_executor = NaiveExecutor(
        pipeline,
        stage=Stage.FIT,
        inputs={
            'X': input_x.meta.inputs['X'],
            'y': input_y.meta.inputs['y'],
        },
        outputs={'probas': average_3.meta.outputs['output']},
    )
    transform_executor = NaiveExecutor(
        pipeline,
        stage=Stage.TRANSFORM,
        inputs={'X': input_x.meta.inputs['X']},
        outputs={
            'probas': average_3.meta.outputs['output'],
            'labels': argmax_3.meta.outputs['output'],
        },
    )
    return pipeline, fit_executor, transform_executor


class FunctionalBlockWrapper:
    def __init__(self, block: BaseBlock, output_name: Optional[str] = None):
        self.block = block
        self.output_name = output_name

    def get_input_slot(self, slot_name: Optional[str] = None):
        if slot_name is None:
            if len(self.block.slots.inputs) == 1:
                return list(self.block.slots.inputs.values())[0]
            else:
                raise RuntimeError('Block has more than one input (please, specify it)')
        return self.block.meta.inputs[slot_name]

    def get_output_slot(self) -> BlockOutputSlot:
        if self.output_name is None:
            if len(self.block.slots.outputs) == 1:
                return list(self.block.slots.outputs.values())[0]
            else:
                raise RuntimeError('Block has more than one output')
        return self.block.slots.outputs[self.output_name]

    def __getitem__(self, output_name: str):
        return FunctionalBlockWrapper(self.block, output_name=output_name)


class FunctionalBuilder:
    def __init__(self):
        self.nodes = []
        self.connections = []

    def __getattr__(self, name: str) -> Callable:
        block_name = name + 'Block'
        block_cls = globals().get(block_name, None)
        if block_cls is None:
            raise ValueError(f'Wrong block class: {name} ({block_name} not found)')
        return self._get_block_init(block_cls)

    def _get_block_init(self, block_cls: Callable) -> Callable:
        def block_init(*args, **kwargs):
            block = block_cls(*args, **kwargs)
            self.nodes.append(block)

            def placeholder_fn(*pfn_args, **pfn_kwargs):
                assert len(pfn_args) == 0, "Only kwargs are supported"
                for input_name, input_block_wrapper in pfn_kwargs.items():
                    self.connections.append(
                        Connection(
                            src=input_block_wrapper.get_output_slot(),
                            dst=block.slots.inputs[input_name],
                        )
                    )
                return FunctionalBlockWrapper(block)

            return placeholder_fn

        return block_init

    def new(self, block_cls: Callable, *args, **kwargs) -> Callable:
        return self._get_block_init(block_cls)(*args, **kwargs)

    @property
    def pipeline(self) -> BasePipeline:
        return BasePipeline(self.nodes, self.connections)


def make_deep_forest_functional_multi_grained_scanning():
    b = FunctionalBuilder()
    X, y = b.Input()(), b.TargetInput()()
    ms = b.new(MultiGrainedScanningBlock, windows=[2], stride=1, random_state=45)(X=X, y=y)
    rf_1 = b.RFC(seed=42)(X=ms, y=y)
    et_1 = b.ETC(seed=42)(X=ms, y=y)
    concat_1 = b.Concat(['ms', 'rf_1', 'et_1'])(ms=ms, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(seed=42)(X=concat_1, y=y)
    et_2 = b.ETC(seed=42)(X=concat_1, y=y)
    concat_2 = b.Concat(['ms', 'rf_2', 'et_2'])(ms=ms, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC(seed=42)(X=concat_2, y=y)
    et_3 = b.ETC(seed=42)(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=rf_1)
    argmax_3 = b.Argmax(axis=1)(X=rf_1)
    #
    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)

    fit_executor = NaiveExecutor(
        b.pipeline,
        stage=Stage.FIT,
        inputs={
            'X': X.get_input_slot(),
            'y': y.get_input_slot(),
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'rf_1_roc-auc': rf_1_roc_auc.get_output_slot()
            # 'roc-auc': roc_auc.get_output_slot(),
        },
    )
    transform_executor = NaiveExecutor(
        b.pipeline,
        stage=Stage.TRANSFORM,
        inputs={
            'X': X.get_input_slot()
        },
        outputs={
            'probas': average_3.get_output_slot(),
            'labels': argmax_3.get_output_slot(),
        },
    )
    return b.pipeline, fit_executor, transform_executor


def make_deep_forest_layer(b, **inputs):
    rf = b.RFC(seed=42)(**inputs)
    et = b.ETC(seed=42)(**inputs)
    stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
    average = b.Average(axis=1)(X=stack)
    return average


def main():
    _, fit_executor, transform_executor = make_deep_forest_functional_multi_grained_scanning()
    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': train_X, 'y': train_y})
    print("Fit successful")
    train_result = transform_executor({'X': train_X})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'], train_result['probas']))
    test_result = transform_executor({'X': test_X})
    print(train_result.keys())
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas']))


if __name__ == "__main__":
    main()

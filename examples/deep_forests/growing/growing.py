"""Example of simple dynamically growing Deep Forest.

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Mapping, Sequence
from bosk import Data
from bosk.executor.base import BaseExecutor
from bosk.executor.handlers import SimpleBlockHandler, InputSlotHandler
from bosk.slot import BlockInputSlot, BlockOutputSlot
import numpy as np
from bosk.block.base import BaseBlock
from bosk.pipeline.connection import Connection
from bosk.pipeline import BasePipeline
from bosk.pipeline.dynamic import BaseDynamicPipeline
from bosk.executor.naive import NaiveExecutor
from bosk.stages import Stage
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import logging

from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import (
    ConcatBlock, StackBlock, AverageBlock, ArgmaxBlock
)
from bosk.block.base import BlockInputData, TransformOutputData
from bosk.block.zoo.metrics import RocAucBlock
from bosk.block.meta import make_simple_meta
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder, FunctionalBlockWrapper


class InputEmbeddingBlock(BaseBlock):
    meta = make_simple_meta(['X'], ['output'])

    def __init__(self):
        super().__init__()

    def fit(self, _inputs: BlockInputData) -> 'InputBlock':
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        return {'output': inputs['X']}


def make_df_layer(input_x_block: InputBlock,
                  input_embedding_block: BaseBlock,
                  input_y_block: TargetInputBlock):
    b = FunctionalPipelineBuilder()
    input_x = b.wrap(input_x_block)()
    input_embedding = b.wrap(input_embedding_block)()
    input_y = b.wrap(input_y_block)()
    rf = b.RFC()
    et = b.ETC()
    stack = b.Stack(['X_0', 'X_1'], axis=1)(
        X_0=rf(X=input_embedding, y=input_y),
        X_1=et(X=input_embedding, y=input_y),
    )
    average = b.Average(axis=1)(X=stack)
    concat = b.Concat(['original', 'transformed'], axis=1)(
        original=input_x,
        transformed=average,
    )
    roc_auc = b.RocAuc()(
        pred_probas=average,
        gt_y=input_y,
    )
    return b.pipeline, concat.block, average.block, roc_auc.block


def make_deep_forest():
    input_x = InputEmbeddingBlock()
    input_y = TargetInputBlock()

    n_layers = 5
    cur_embedding = input_x  # initial embedding is raw input
    dynamic_pipeline = BaseDynamicPipeline([input_x, input_y], [])
    for _ in range(n_layers):
        layer_pipeline, layer_concat, layer_average, roc_auc = make_df_layer(
            input_x,
            cur_embedding,
            input_y
        )
        dynamic_pipeline.extend(layer_pipeline)
        cur_embedding = layer_concat

    argmax = ArgmaxBlock(axis=1)
    end_pipeline = BasePipeline(
        nodes=[
            argmax
        ],
        connections=[
            Connection(layer_average.slots.outputs['output'], argmax.slots.inputs['X']),
        ]
    )
    dynamic_pipeline.extend(end_pipeline)

    fit_executor = NaiveExecutor(
        dynamic_pipeline,
        InputSlotHandler(Stage.FIT),
        SimpleBlockHandler(Stage.FIT),
        stage=Stage.FIT,
        inputs={
            'X': input_x.slots.inputs['X'],
            'y': input_y.slots.inputs['y'],
        },
        outputs={'probas': layer_average.slots.outputs['output']},
    )
    transform_executor = NaiveExecutor(
        dynamic_pipeline,
        InputSlotHandler(Stage.TRANSFORM),
        SimpleBlockHandler(Stage.TRANSFORM),
        stage=Stage.TRANSFORM,
        inputs={'X': input_x.slots.inputs['X']},
        outputs={
            'probas': layer_average.slots.outputs['output'],
            'labels': argmax.slots.outputs['output'],
        },
    )
    return dynamic_pipeline, fit_executor, transform_executor


@dataclass
class GrowingResult:
    pipeline: BasePipeline
    fit_inputs: Mapping[str, BlockInputSlot]
    transform_inputs: Mapping[str, BlockInputSlot]
    fit_outputs: Mapping[str, BlockOutputSlot]
    transform_outputs: Mapping[str, BlockOutputSlot]
    connections_extension: List[Connection]


class BaseGrower(ABC):
    @abstractmethod
    def need_grow(self, cur_output_data: Mapping[str, Data], transform_executor: BaseExecutor) -> bool:
        ...

    @abstractmethod
    def grow(self, input_outputs: Mapping[str, BlockOutputSlot]) -> GrowingResult:
        ...


class DynamicExecutor:
    def __init__(self, pipeline: BaseDynamicPipeline,
                 fit_inputs: Mapping[str, BlockInputSlot | Sequence[BlockInputSlot]],
                 fit_outputs: Mapping[str, BlockOutputSlot],
                 transform_inputs: Mapping[str, BlockInputSlot | Sequence[BlockInputSlot]],
                 transform_outputs: Mapping[str, BlockOutputSlot],
                 grower: BaseGrower,
                 executor_cls=NaiveExecutor):
        self._pipeline = pipeline
        self._fit_inputs = fit_inputs
        self._fit_outputs = fit_outputs
        self._transform_inputs = transform_inputs
        self._transform_outputs = transform_outputs
        self._grower = grower
        self._executor_cls = executor_cls
        self._transform_executor = None

    def fit(self, input_data: BlockInputData) -> Mapping[str, Data]:
        cur_pipeline = self._pipeline
        cur_fit_inputs = self._fit_inputs
        cur_fit_outputs = self._fit_outputs
        cur_transform_inputs = self._transform_inputs
        cur_transform_outputs = self._transform_outputs
        cur_input_data = input_data
        while True:
            fit_executor = self._executor_cls(
                cur_pipeline,
                InputSlotHandler(Stage.FIT),
                SimpleBlockHandler(Stage.FIT),
                stage=Stage.FIT,
                inputs=cur_fit_inputs,
                outputs=cur_fit_outputs,
            )
            transform_executor = self._executor_cls(
                cur_pipeline,
                InputSlotHandler(Stage.TRANSFORM),
                SimpleBlockHandler(Stage.TRANSFORM),
                stage=Stage.TRANSFORM,
                inputs=cur_transform_inputs,
                outputs=cur_transform_outputs,
            )
            cur_output_data = fit_executor(cur_input_data)
            if not self._grower.need_grow(cur_output_data, transform_executor):
                break

            growing_result = self._grower.grow(
                cur_fit_outputs,
            )
            cur_fit_inputs = growing_result.fit_inputs
            cur_transform_inputs = growing_result.transform_inputs
            cur_fit_outputs = growing_result.fit_outputs
            cur_transform_outputs = growing_result.transform_outputs

            cur_pipeline = growing_result.pipeline
            cur_input_data = cur_output_data
            self._pipeline.extend(cur_pipeline, growing_result.connections_extension)

        self._fit_outputs = cur_fit_outputs
        self._transform_outputs = cur_transform_outputs
        # prepare transform executor for `transform` method
        self._transform_executor = self._executor_cls(
            self._pipeline,
            InputSlotHandler(Stage.TRANSFORM),
            SimpleBlockHandler(Stage.TRANSFORM),
            stage=Stage.TRANSFORM,
            inputs=self._transform_inputs,
            outputs=self._transform_outputs,
        )
        return cur_output_data

    def transform(self, input_data: BlockInputData) -> Mapping[str, Data]:
        return self._transform_executor(input_data)


class EmptyGrower(BaseGrower):
    def need_grow(self, *args, **kwargs) -> bool:
        return False

    def grow(self, *args, **kwargs) -> GrowingResult:
        raise NotImplementedError()


class EarlyStoppingGrower(BaseGrower):
    def __init__(self, val_X, val_y, min_layers: int = 1, max_layers: int = 5):
        self.min_layers = min_layers
        self.max_layers = max_layers
        self._layer_num = 0
        self._prev_val_outputs = None
        self._prev_val_score = None
        self.val_X = val_X
        self.val_y = val_y

    def need_grow(self, cur_output_data: Mapping[str, Data], transform_executor: BaseExecutor) -> bool:
        if self._layer_num >= self.max_layers:
            return False
        if self._layer_num == 0:
            cur_input = {
                'X': self.val_X,
            }
        else:
            cur_input = {
                'X': self._prev_val_outputs['X'],
                'concat': self._prev_val_outputs['concat']
            }
        self._prev_val_outputs = transform_executor(cur_input)
        score = roc_auc_score(self.val_y, self._prev_val_outputs['probas'][:, 1])
        prev_score = self._prev_val_score
        self._prev_val_score = score
        if prev_score is None:
            return True
        if self._layer_num < self.min_layers:
            return True
        if score < prev_score:
            logging.info(
                'Stopping after %d iterations. Val scores: %lf -> %lf',
                self._layer_num,
                prev_score,
                score
            )
            return False
        return True

    def grow(self, prev_layer_outputs: Mapping[str, BlockOutputSlot]) -> GrowingResult:
        self._layer_num += 1

        cur_input_x = InputEmbeddingBlock()
        cur_input_embedding = InputEmbeddingBlock()
        cur_input_y = TargetInputBlock()
        cur_layer_pipeline, cur_layer_concat, cur_layer_average, cur_layer_roc_auc = make_df_layer(
            cur_input_x,
            cur_input_embedding,
            cur_input_y
        )
        connections_extension = [
            Connection(prev_layer_outputs['X'], cur_input_x.slots.inputs['X']),
            Connection(prev_layer_outputs['y'], cur_input_y.slots.inputs['y']),
            Connection(prev_layer_outputs['concat'], cur_input_embedding.slots.inputs['X']),
        ]

        return GrowingResult(
            pipeline=cur_layer_pipeline,
            fit_inputs={
                'X': cur_input_x.slots.inputs['X'],
                'y': cur_input_y.slots.inputs['y'],
                'concat': cur_input_embedding.slots.inputs['X'],
            },
            transform_inputs={
                'X': cur_input_x.slots.inputs['X'],
                'concat': cur_input_embedding.slots.inputs['X'],
            },
            fit_outputs={
                'X': cur_input_x.slots.outputs['output'],
                'y': cur_input_y.slots.outputs['y'],
                'concat': cur_layer_concat.slots.outputs['output'],
                'probas': cur_layer_average.slots.outputs['output'],
                'roc_auc': cur_layer_roc_auc.slots.outputs['roc-auc'],
            },
            transform_outputs={
                'X': cur_input_x.slots.outputs['output'],
                'concat': cur_layer_concat.slots.outputs['output'],
                'probas': cur_layer_average.slots.outputs['output'],
            },
            connections_extension=connections_extension,
        )


def make_deep_forest_with_early_stopping(val_X, val_y) -> DynamicExecutor:
    input_x = InputEmbeddingBlock()
    input_y = TargetInputBlock()

    cur_embedding = input_x  # initial embedding is raw input
    dynamic_pipeline = BaseDynamicPipeline([input_x, input_y], [])
    layer_pipeline, layer_concat, layer_average, roc_auc = make_df_layer(
        input_x,
        cur_embedding,
        input_y
    )
    dynamic_pipeline.extend(layer_pipeline)
    cur_embedding = layer_concat

    dynamic_executor = DynamicExecutor(
        dynamic_pipeline,
        fit_inputs={
            'X': input_x.slots.inputs['X'],
            'y': input_y.slots.inputs['y'],
        },
        fit_outputs={
            'X': input_x.slots.outputs['output'],
            'y': input_y.slots.outputs['y'],
            'concat': layer_concat.slots.outputs['output'],
            'probas': layer_average.slots.outputs['output'],
            'roc_auc': roc_auc.slots.outputs['roc-auc'],
        },
        transform_inputs={'X': input_x.slots.inputs['X']},
        transform_outputs={
            'X': input_x.slots.outputs['output'],
            'concat': layer_concat.slots.outputs['output'],
            'probas': layer_average.slots.outputs['output'],
        },
        # grower=EmptyGrower(),
        grower=EarlyStoppingGrower(val_X, val_y, min_layers=2, max_layers=5),
        executor_cls=NaiveExecutor,
    )

    return dynamic_executor


def main():
    logging.basicConfig(
        level=logging.INFO
    )
    # _pipeline, fit_executor, transform_executor = make_deep_forest()
    all_X, all_y = make_moons(noise=0.5)
    tv_X, test_X, tv_y, test_y = train_test_split(all_X, all_y, test_size=0.2)
    train_X, val_X, train_y, val_y = train_test_split(tv_X, tv_y, test_size=0.2)
    dynamic_executor = make_deep_forest_with_early_stopping(val_X, val_y)

    # fit_result = fit_executor({'X': train_X, 'y': train_y})
    fit_result = dynamic_executor.fit({'X': train_X, 'y': train_y})
    print("Fit successful")
    # train_result = transform_executor({'X': train_X})
    train_result = dynamic_executor.transform({'X': train_X})
    print("Fit probas == probas on train:", np.allclose(fit_result['probas'], train_result['probas']))
    # test_result = transform_executor({'X': test_X})
    test_result = dynamic_executor.transform({'X': test_X})
    print(train_result.keys())
    print("Train ROC-AUC:", roc_auc_score(train_y, train_result['probas'][:, 1]))
    print("Test ROC-AUC:", roc_auc_score(test_y, test_result['probas'][:, 1]))


if __name__ == "__main__":
    main()

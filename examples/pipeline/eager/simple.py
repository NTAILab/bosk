"""Example of Eager evaluation: applying fit and transform stages during pipeline construnction.

"""
import json
from bosk.data import CPUData
from bosk.stages import Stage

import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bosk.executor.block import (
    BaseBlockExecutor,
    DefaultBlockExecutor,
    InputSlotToDataMapping,
    BlockOutputData,
    BaseData,
)
from bosk.block.base import BaseInputBlock
from bosk.executor.recursive import RecursiveExecutor

from sklearn.metrics import roc_auc_score
from bosk.block import BaseBlock

from bosk.block.functional import FunctionalBlockWrapper
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder, BaseBlockClassRepository
from bosk.pipeline.connection import Connection

from typing import Callable, Optional, Mapping


class EagerBlockWrapper(FunctionalBlockWrapper):
    def __init__(self, block: BaseBlock, executor: BaseBlockExecutor,
                 output_name: Optional[str] = None):
        super().__init__(block, output_name=output_name)
        self.executor = executor
        self.fit_output_values: Optional[Mapping[str, BaseData]] = None

    def execute(self, block_input_mapping: InputSlotToDataMapping) -> BlockOutputData:
        """Execute the underlying block with the given inputs and store in the wrapper state.

        Args:
            inputs: Block inputs.

        """
        assert self.fit_output_values is None, 'Cannot fit the eager block twice'
        block_output_data = self.executor.execute_block(
            stage=Stage.FIT,
            block=self.block,
            block_input_mapping=block_input_mapping,
        )
        self.fit_output_values = block_output_data
        return block_output_data

    def get_output_data(self) -> BlockOutputData:
        assert self.fit_output_values is not None
        return self.fit_output_values[self.get_output_slot()]

    def __getitem__(self, output_name: str) -> 'EagerBlockWrapper':
        eager_block = EagerBlockWrapper(self.block, output_name=output_name)
        eager_block.fit_output_values = self.fit_output_values
        return eager_block


class EagerPipelineBuilder(FunctionalPipelineBuilder):
    def __init__(self, block_executor: BaseBlockExecutor,
                 block_repo: Optional[BaseBlockClassRepository] = None):
        super().__init__(block_repo)
        self.block_executor = block_executor

    def _make_placeholder_fn(self, block: BaseBlock) -> Callable[..., EagerBlockWrapper]:
        def placeholder_fn(*pfn_args, **pfn_kwargs) -> EagerBlockWrapper:
            if len(pfn_args) > 0:
                assert len(pfn_kwargs) == 0, \
                    'Either unnamed or named arguments can be used, but not at the same time'
                assert len(pfn_args) == 1, \
                    'Only one unnamed argument is supported (we can infer name only in this case)'
                assert isinstance(block, BaseInputBlock)
                pfn_kwargs = {
                    block.get_single_input().meta.name: pfn_args[0]
                }
            block_input_mapping: InputSlotToDataMapping = dict()
            for input_name, input_block_wrapper_or_data in pfn_kwargs.items():
                block_input = block.slots.inputs[input_name]
                if isinstance(input_block_wrapper_or_data, EagerBlockWrapper):
                    self._connections.append(
                        Connection(
                            src=input_block_wrapper_or_data.get_output_slot(),
                            dst=block_input,
                        )
                    )
                    block_input_mapping[block_input] = input_block_wrapper_or_data.get_output_data()
                elif isinstance(input_block_wrapper_or_data, BaseData):
                    block_input_mapping[block_input] = input_block_wrapper_or_data
                else:
                    raise ValueError(
                        f'Wrong placeholder input type: {type(input_block_wrapper_or_data)}'
                    )
            eager_block = EagerBlockWrapper(block, executor=self.block_executor)
            if len(block_input_mapping) > 0:
                eager_block.execute(block_input_mapping)
            return eager_block

        return placeholder_fn


def main():
    # prepare data first
    all_X, all_y = load_breast_cancer(return_X_y=True)
    train_X, test_X, train_y, test_y = train_test_split(all_X, all_y, test_size=0.2, random_state=42)

    executor_cls = RecursiveExecutor
    block_executor = DefaultBlockExecutor()
    forest_params = dict()

    b = EagerPipelineBuilder(block_executor)
    X, y = b.Input()(CPUData(train_X)), b.TargetInput()(CPUData(train_y))
    # alternative ways to define input blocks:
    # X, y = b.Input()(X=CPUData(train_X)), b.TargetInput()(y=CPUData(train_y))
    # or without input data:
    # X, y = b.Input()(), b.TargetInput()()
    # X.execute({X.block.get_single_input(): CPUData(train_X)})
    # y.execute({y.block.get_single_input(): CPUData(train_y)})

    rf_1 = b.RFC(random_state=42, **forest_params)(X=X, y=y)
    et_1 = b.ETC(random_state=42, **forest_params)(X=X, y=y)
    concat_1 = b.Concat(['X', 'rf_1', 'et_1'])(X=X, rf_1=rf_1, et_1=et_1)
    rf_2 = b.RFC(random_state=42, **forest_params)(X=concat_1, y=y)
    et_2 = b.ETC(random_state=42, **forest_params)(X=concat_1, y=y)
    concat_2 = b.Concat(['X', 'rf_2', 'et_2'])(X=X, rf_2=rf_2, et_2=et_2)
    rf_3 = b.RFC(random_state=42, **forest_params)(X=concat_2, y=y)
    et_3 = b.ETC(random_state=42, **forest_params)(X=concat_2, y=y)
    stack_3 = b.Stack(['rf_3', 'et_3'], axis=1)(rf_3=rf_3, et_3=et_3)
    average_3 = b.Average(axis=1)(X=stack_3)
    argmax_3 = b.Argmax(axis=1)(X=average_3)

    rf_1_roc_auc = b.RocAuc()(gt_y=y, pred_probas=rf_1)
    roc_auc = b.RocAuc()(gt_y=y, pred_probas=average_3)

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
    main()

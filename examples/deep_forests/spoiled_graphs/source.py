"""Example of the misconnected graph.
"""

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.handlers import SimpleExecutionStrategy, InputSlotStrategy
from bosk.stages import Stage
from bosk.block.zoo.models.classification import RFCBlock, ETCBlock
from bosk.block.zoo.data_conversion import ConcatBlock, AverageBlock, ArgmaxBlock, StackBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.block.zoo.metrics import RocAucBlock
from bosk.executor.topological import TopologicalExecutor

# one vital connection of this graph is forgotten
# only one output rf_1_roc-auc remains
def make_rf_lost_connection(executor, **ex_kw):
    input_x = InputBlock()
    input_y = TargetInputBlock()
    rf_1 = RFCBlock(random_state=42)
    et_1 = ETCBlock(random_state=42)
    concat_1 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_2 = RFCBlock(random_state=42)
    et_2 = ETCBlock(random_state=42)
    concat_2 = ConcatBlock(['X_0', 'X_1'], axis=1)
    rf_3 = RFCBlock(random_state=42)
    et_3 = ETCBlock(random_state=42)
    stack_3 = StackBlock(['X_0', 'X_1'], axis=1)
    average_3 = AverageBlock(axis=1)
    argmax_3 = ArgmaxBlock(axis=1)
    roc_auc = RocAucBlock()
    roc_auc_rf_1 = RocAucBlock()
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
            roc_auc,
            roc_auc_rf_1
        ],
        connections=[
            # input X
            Connection(input_x.slots.outputs['X'], rf_1.slots.inputs['X']),
            Connection(input_x.slots.outputs['X'], et_1.slots.inputs['X']),
            # input y
            Connection(input_y.slots.outputs['y'], rf_1.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], et_1.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], rf_2.slots.inputs['y']),
            # Connection(input_y.slots.outputs['y'], et_2.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], rf_3.slots.inputs['y']),
            Connection(input_y.slots.outputs['y'], et_3.slots.inputs['y']),
            # layers connection
            Connection(rf_1.slots.outputs['output'], concat_1.slots.inputs['X_0']),
            Connection(et_1.slots.outputs['output'], concat_1.slots.inputs['X_1']),
            Connection(concat_1.slots.outputs['output'], rf_2.slots.inputs['X']),
            Connection(concat_1.slots.outputs['output'], et_2.slots.inputs['X']),
            Connection(rf_2.slots.outputs['output'], concat_2.slots.inputs['X_0']),
            Connection(et_2.slots.outputs['output'], concat_2.slots.inputs['X_1']),
            Connection(concat_2.slots.outputs['output'], rf_3.slots.inputs['X']),
            Connection(concat_2.slots.outputs['output'], et_3.slots.inputs['X']),
            Connection(rf_3.slots.outputs['output'], stack_3.slots.inputs['X_0']),
            Connection(et_3.slots.outputs['output'], stack_3.slots.inputs['X_1']),
            Connection(stack_3.slots.outputs['output'], average_3.slots.inputs['X']),
            Connection(average_3.slots.outputs['output'], argmax_3.slots.inputs['X']),
            Connection(average_3.slots.outputs['output'], roc_auc.slots.inputs['pred_probas']),
            Connection(input_y.slots.outputs['y'], roc_auc.slots.inputs['gt_y']),
            Connection(rf_1.slots.outputs['output'], roc_auc_rf_1.slots.inputs['pred_probas']),
            Connection(input_y.slots.outputs['y'], roc_auc_rf_1.slots.inputs['gt_y']),
        ],
        inputs={
            'X': input_x.slots.inputs['X'],
            'y': input_y.slots.inputs['y'],
        },
        outputs={
            'probas': average_3.slots.outputs['output'],
            'rf_1_roc-auc': roc_auc_rf_1.slots.outputs['roc-auc'],
            'roc-auc': roc_auc.slots.outputs['roc-auc'],
            'labels': argmax_3.slots.outputs['output']
        }
    )

    fit_executor = executor(
        pipeline,
        InputSlotStrategy(Stage.FIT),
        SimpleExecutionStrategy(Stage.FIT),
        stage=Stage.FIT,
        inputs={
            'X': input_x.slots.inputs['X'],
            'y': input_y.slots.inputs['y'],
        },
        outputs={
            'probas': average_3.slots.outputs['output'],
            'rf_1_roc-auc': roc_auc_rf_1.slots.outputs['roc-auc'],
            'roc-auc': roc_auc.slots.outputs['roc-auc'],
        },
        **ex_kw
    )
    transform_executor = executor(
        pipeline,
        InputSlotStrategy(Stage.TRANSFORM),
        SimpleExecutionStrategy(Stage.TRANSFORM),
        stage=Stage.TRANSFORM,
        inputs={'X': input_x.slots.inputs['X']},
        outputs={
            'probas': average_3.slots.outputs['output'],
            'labels': argmax_3.slots.outputs['output'],
        },
        **ex_kw
    )
    return fit_executor, transform_executor

def main():
    executor_class = TopologicalExecutor
    fit_executor, transform_executor = make_rf_lost_connection(executor_class)

    all_X, all_y = make_moons(noise=0.5, random_state=42)
    train_X, test_X, train_y, _ = train_test_split(all_X, all_y, test_size=0.2, random_state=42)
    fit_result = fit_executor({'X': train_X, 'y': train_y})
    print('fit result contains next data:')
    for key, val in fit_result.items():
        print(key, val)
    test_result = transform_executor({'X': test_X})
    print('test result contains next data:')
    for key, val in test_result.items():
        print(key, val)


if __name__ == "__main__":
    main()
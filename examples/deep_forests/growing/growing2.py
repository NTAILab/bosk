from abc import ABC, abstractmethod
from typing import List, Mapping, Optional, Set, Type, Any, Sequence, Dict
from dataclasses import dataclass, field
from bosk import Data
from bosk.executor.base import BaseExecutor
from bosk.executor.handlers import SimpleExecutionStrategy, InputSlotStrategy
from bosk.painter.topological import TopologicalPainter
from bosk.slot import BlockInputSlot, BlockOutputSlot
from bosk.block.base import BaseBlock
from bosk.pipeline.connection import Connection
from bosk.pipeline import BasePipeline
from bosk.executor.topological import TopologicalExecutor
from bosk.stages import Stage
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.pipeline.builder.functional import FunctionalPipelineBuilder

import warnings
import copy
from collections import defaultdict


class StepwisePipeline():
    """Structure that provides possibility to build the pipeline
    step by step. You select input blocks and output names of the final pipeline
    and then merge new parts into the global result. 
    
    The main idea is to use same names of the inputs and outputs which should be
    connected between steps.
    
    The common input blocks will try to connect with every added step.
    
    The global output slots will be chosen by the common output names and the
    preference will be given to the last added output slot with the corresponding name.

    """
    def __init__(self, input_blocks: Mapping[str, BaseBlock], outputs: Set[str]):
        for inp_block in input_blocks.values():
            assert len(inp_block.slots.inputs) == 1, "Input blocks must contain only 1 input slot"
            assert len(inp_block.slots.outputs) == 1, "Input blocks must contain only 1 output slot"
        self.common_inputs = input_blocks
        self.common_outputs = outputs
        self.cur_nodes: List[BaseBlock] = list(input_blocks.values())
        self.cur_internal_conns: List[Connection] = []
        self.cur_inp_conns: Mapping[str, List[BlockInputSlot]] = defaultdict(list)
        self.cur_outputs: Mapping[str, BlockOutputSlot] = dict()

    def merge_new_step(self, pipeline: BasePipeline, f_check_in_out: bool = True):
        if f_check_in_out:
            for inp in self.common_inputs:
                if inp not in pipeline.inputs:
                    warnings.warn(f'Unable to find common input "{inp}" in the new step')
            for out in self.common_outputs:
                if out not in pipeline.outputs:
                    warnings.warn(f'Unable to find common output "{out}" in the new step')
        for inp_name in self.common_inputs:
            inp_slot = pipeline.inputs.get(inp_name, None)
            if inp_slot is not None:
                self.cur_inp_conns[inp_name].append(inp_slot)
        for inp_name, inp_slot in pipeline.inputs.items():
            out_slot = self.cur_outputs.get(inp_name, None)
            if out_slot is not None:
                self.cur_internal_conns.append(Connection(out_slot, inp_slot))
        for out_name in self.common_outputs:
            out_slot = pipeline.outputs.get(out_name, None)
            if out_slot is not None:
                self.cur_outputs[out_name] = out_slot
        self.cur_internal_conns.extend(pipeline.connections)
        self.cur_nodes.extend(pipeline.nodes)

    def build_base_pipeline(self):
        inp_conns = []
        for inp_name, inp_block in self.common_inputs.items():
            out_slot = next(iter(inp_block.slots.outputs.values()))
            for inp_slot in self.cur_inp_conns[inp_name]:
                inp_conns.append(Connection(out_slot, inp_slot))
        inputs = {key: next(iter(block.slots.inputs.values())) for key, block in self.common_inputs.items()}
        return BasePipeline(nodes=self.cur_nodes, connections=self.cur_internal_conns + inp_conns,
                        inputs=inputs, outputs=self.cur_outputs)


class BaseGrowingStrategy(ABC):
    """Class that defines the growing strategy of the deep forest.
    It computes metrics in the :meth:`need_grow` method and decides
    if it is needed to make further grow.
    """
    @abstractmethod
    def need_grow(self, labels: Optional[Mapping[str, Data]] = None,
                transform_results: Optional[Mapping[str, Data]] = None,
                fit_results: Optional[Mapping[str, Data]] = None) -> bool:
        ...
    
    @abstractmethod
    def get_details(self) -> Mapping[str, Any]:
        ...

class LayerFactoryBase(ABC):
    @abstractmethod
    def get_new_layer(self) -> BasePipeline:
        ...

@dataclass
class EarlyStoppingStrategy(BaseGrowingStrategy):
    max_iter_num: int 
    patience: int # todo
    _cur_iter: int = field(default=0, init=False)

    def need_grow(self, labels: Optional[Mapping[str, Data]] = None,
                transform_results: Optional[Mapping[str, Data]] = None,
                fit_results: Optional[Mapping[str, Data]] = None) -> bool:
        self._cur_iter += 1
        return self._cur_iter < self.max_iter_num
    
    def get_details(self) -> Mapping[str, Any]:
        return {'iterations remain': self.max_iter_num - self._cur_iter}

class ROCAUCStrategy(BaseGrowingStrategy):
    def __init__(self):
        self.last_score = 0

    def need_grow(self, labels: Optional[Mapping[str, Data]] = None,
                transform_results: Optional[Mapping[str, Data]] = None,
                fit_results: Optional[Mapping[str, Data]] = None) -> bool:
        assert labels is not None
        if transform_results is None:
            return True
        current_score = roc_auc_score(labels['labels'], transform_results['probas'][:, 1])
        result = self.last_score < current_score
        self.last_score = current_score
        return result
    
    def get_details(self) -> Mapping[str, Any]:
        return {'roc-auc test score': self.last_score}

class ComplexGrowingStrategy(BaseGrowingStrategy):
    def __init__(self, strategies: Sequence[BaseGrowingStrategy]) -> None:
        self.strat_seq = copy.copy(strategies)
    
    def need_grow(self, labels: Optional[Mapping[str, Data]] = None,
                transform_results: Optional[Mapping[str, Data]] = None,
                fit_results: Optional[Mapping[str, Data]] = None) -> bool:
        return all(map(lambda strat: strat.need_grow(labels, transform_results, fit_results), self.strat_seq))
    
    def get_details(self) -> Mapping[str, Any]:
        res = dict()
        for i, strat in enumerate(self.strat_seq):
            cur_details = strat.get_details()
            for key, val in cur_details.items():
                if key in res:
                    res[f'{key}_{i + 1}'] = val
                else:
                    res[key] = val
        return res

class SimpleDFLayerFactory(LayerFactoryBase):
    def __init__(self):
        self.called = 0

    def get_new_layer(self) -> BasePipeline:
        b = FunctionalPipelineBuilder()
        if self.called == 0:
            # boundary case
            # another solution is to rewrite the ConcatBlock
            input_concat = b.Input()()
        else:
            input_concat = b.Concat(['X', 'predict'])()
        input_y = b.TargetInput()()
        rf = b.RFC()(X=input_concat, y=input_y)
        et = b.ETC()(X=input_concat, y=input_y)
        stack = b.Stack(['rf', 'et'], axis=1)(rf=rf, et=et)
        prob = b.Average(axis=1)(X=stack)
        labels = b.Argmax(axis=1)(X=prob)
        roc_auc = b.RocAuc()(gt_y=input_y, pred_probas=prob)
        inputs = {
            'X': input_concat.get_input_slot('X'),
            'y': input_y,
        }
        if self.called > 0:
            inputs['probas'] = input_concat.get_input_slot('predict')
        outputs = {
            'probas': prob,
            'labels': labels,
            'roc-auc': roc_auc
        }
        self.called += 1
        return b.build(inputs, outputs)

class FitCallback(ABC):
    @abstractmethod
    def __call__(self, df_fit_output: Mapping[str, Data]) -> None:
        ...

# this test is specically made to test unnecessary internal output
class ROCAUCCallback(FitCallback):
    def __call__(self, df_fit_output: Mapping[str, Data]) -> None:
        print('Fit roc-auc score:', df_fit_output['roc-auc'])

class SimpleGrowingFitter():
    def __init__(self, pipeline: StepwisePipeline,
                layer_factory: LayerFactoryBase, growing_strategy: BaseGrowingStrategy,
                executor_cls: Type[BaseExecutor], exec_kw = {},
                fit_callback: Optional[FitCallback] = None):
        self.growing_pipeline = pipeline
        self.layer_factory = layer_factory
        self.strategy = growing_strategy
        self.exec_cls = executor_cls
        self.exec_kw = exec_kw
        self.is_fitted = False
        self.fit_callback = fit_callback

    def _print_log(self):
        grow_details = self.strategy.get_details()
        print('Testing log:')
        for key, val in grow_details.items():
            print(f'\t{key}: {val}')

    def fit(self, train_data: Dict[str, Data], val_data: Dict[str, Data], labels: Dict[str, Data]) -> None:
        input_fit_data = copy.copy(train_data)
        input_test_data = copy.copy(val_data)
        fit_output = None
        test_output = None
        i = 1
        print('Beginning the fit process')
        while True:
            print(f'--- Iteration {i} ---')
            new_layer = self.layer_factory.get_new_layer()
            fit_exec = self.exec_cls(new_layer,
                InputSlotStrategy(Stage.FIT),
                SimpleExecutionStrategy(Stage.FIT),
                stage=Stage.FIT, **self.exec_kw)
            fit_output = fit_exec(input_fit_data)
            if self.fit_callback is not None:
                self.fit_callback(fit_output)
            tf_exec = self.exec_cls(new_layer,
                InputSlotStrategy(Stage.TRANSFORM),
                SimpleExecutionStrategy(Stage.TRANSFORM),
                stage=Stage.TRANSFORM, **self.exec_kw)
            test_output = tf_exec(input_test_data)
            f_need_continue = self.strategy.need_grow(labels, test_output, fit_output)
            self._print_log()
            if not f_need_continue:
                print('Training is stopped')
                break
            print('Layer is added')
            self.growing_pipeline.merge_new_step(new_layer)
            input_fit_data.update(fit_output)
            input_test_data.update(test_output)
            i += 1
        self.is_fitted = True
        
    def get_pipeline(self):
        assert self.is_fitted, "You must fit the deep forest first"
        return self.growing_pipeline.build_base_pipeline()
    
    def get_transform_executor(self):
        assert self.is_fitted, "You must fit the deep forest first"
        return self.exec_cls(
            self.growing_pipeline.build_base_pipeline(),
            InputSlotStrategy(Stage.TRANSFORM),
            SimpleExecutionStrategy(Stage.TRANSFORM),
            stage=Stage.TRANSFORM,
            inputs=self.growing_pipeline.common_inputs.keys(),
            outputs=self.growing_pipeline.common_outputs,
        )


def main():
    all_X, all_y = make_moons(noise=0.5)
    tv_X, test_X, tv_y, test_y = train_test_split(all_X, all_y, test_size=0.2)
    train_X, val_X, train_y, val_y = train_test_split(tv_X, tv_y, test_size=0.2)
    growing_pipeline = StepwisePipeline(
        {
            'X': InputBlock(),
            'y': TargetInputBlock(),
        },
        {'probas', 'labels'}
    )
    early_stop_strat = EarlyStoppingStrategy(5, 0)
    roc_auc_strat = ROCAUCStrategy()
    final_strat = ComplexGrowingStrategy([early_stop_strat, roc_auc_strat])
    growing_manager = SimpleGrowingFitter(growing_pipeline, SimpleDFLayerFactory(), final_strat,
        TopologicalExecutor, fit_callback=ROCAUCCallback())

    growing_manager.fit({'X': train_X, 'y': train_y}, {'X': val_X}, {'labels': val_y})
    tf_exec = growing_manager.get_transform_executor()
    output = tf_exec({'X': test_X})

    TopologicalPainter().from_executor(tf_exec).render('growing forest.png')
    print("Test ROC-AUC:", roc_auc_score(test_y, output['probas'][:, 1]))

if __name__ == "__main__":
    main()
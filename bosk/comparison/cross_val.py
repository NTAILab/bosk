from .base import BaseComparator, BaseForeignModel
from .metric import BaseMetric
from bosk.executor.base import BaseExecutor
from bosk.data import BaseData
from bosk.pipeline.base import BasePipeline
from bosk.stages import Stage
from bosk.executor.topological import TopologicalExecutor
from bosk.executor.descriptor import HandlingDescriptor
from copy import deepcopy
import numpy as np
from typing import List, Dict
from sklearn.model_selection import BaseCrossValidator
import logging


class CVComparator(BaseComparator):
    """Comparator that uses a cross-validation
    strategy of checking model's performance. You can create
    your own iterator (based on sklearn's `BaseCrossValidator` class)
    that will define indexes, taken in each fold,
    or you can use predefined iterators from the `sklearn`.
    """

    def _get_nested_res_dict(self, metrics: List[BaseMetric]) -> Dict[str, List]:
        nested_res_dict = dict()
        for i, metric in enumerate(metrics):
            name = metric.name
            if name is None:
                name = f'metric_{i}'
            nested_res_dict[name + '_train'] = []
            nested_res_dict[name + '_test'] = []
        return nested_res_dict

    def _write_fold_info_to_dict(self, res_dict, dict_key,
                                 metrics, train_dict, pred_train_dict,
                                 test_dict, pred_test_dict) -> None:
        for i, metric in enumerate(metrics):
            name = metric.name
            if name is None:
                name = f'metric_{i}'
            res_dict[dict_key][name + '_train'].append(
                metric.get_score(train_dict, pred_train_dict))
            res_dict[dict_key][name + '_test'].append(
                metric.get_score(test_dict, pred_test_dict))

    def __init__(self, pipelines: List[BasePipeline], common_part: BasePipeline,
                 foreign_models: List[BaseForeignModel], cv_strat: BaseCrossValidator,
                 exec_cls: BaseExecutor = TopologicalExecutor, random_state: int = None) -> None:
        super().__init__(pipelines, common_part, foreign_models, random_state)
        cv_strat.random_state = random_state
        self.cv_strat = cv_strat
        self.exec_cls = exec_cls

    def get_score(self, data: Dict[str, BaseData], metrics: List[BaseMetric]) -> Dict[str, Dict[str, float]]:
        n = None
        for value in data.values():
            if n is None:
                n = value.data.shape[0]
            else:
                assert value.data.shape[0] == n, "All inputs must have the same number of samples"

        res_dict = self._get_results_dict(self._get_nested_res_dict(metrics))
        idx = np.arange(n)
        for i, (train_idx, test_idx) in enumerate(self.cv_strat.split(idx)):
            logging.info('Processing fold #%i', i)

            # getting fold subsets of data
            train_dict = dict()
            test_dict = dict()
            for key, val in data.items():
                train_dict[key] = val.__class__(val.data[train_idx])
                test_dict[key] = val.__class__(val.data[test_idx])

            # processing the common part
            if self.common_pipeline is None:
                common_train_res = dict()
                common_test_res = common_train_res
            else:
                train_exec = self.exec_cls(self.common_pipeline,
                                           HandlingDescriptor.from_classes(Stage.FIT))
                common_train_res = train_exec(train_dict)
                test_exec = self.exec_cls(self.common_pipeline,
                                          HandlingDescriptor.from_classes(Stage.TRANSFORM))
                common_test_res = test_exec(test_dict)

            for j, cur_pipeline in enumerate(self.optim_pipelines):
                pipeline = deepcopy(cur_pipeline)
                # build personal train dict
                cur_train_dict = dict()
                for key in pipeline.inputs.keys():
                    if key in train_dict:
                        cur_train_dict[key] = train_dict[key]
                    elif key in common_train_res:
                        cur_train_dict[key] = common_train_res[key]
                    else:
                        raise RuntimeError(
                            f"Unable to find '{key}' key neither in the data nor in the common part")

                pip_tr_exec = self.exec_cls(pipeline,
                                            HandlingDescriptor.from_classes(Stage.FIT))
                pip_train_res = pip_tr_exec(cur_train_dict)

                # build personal test dict
                cur_test_dict = dict()
                for key in pipeline.inputs.keys():
                    if key in test_dict:
                        cur_test_dict[key] = test_dict[key]
                    elif key in common_test_res:
                        cur_test_dict[key] = common_test_res[key]
                    else:
                        raise RuntimeError(
                            f"Unable to find '{key}' key neither in the data nor in the common part")

                pip_test_exec = self.exec_cls(pipeline,
                                              HandlingDescriptor.from_classes(Stage.TRANSFORM))
                pip_test_res = pip_test_exec(cur_test_dict)

                self._write_fold_info_to_dict(res_dict, f'pipeline_{j}', metrics,
                                              train_dict, pip_train_res,
                                              test_dict, pip_test_res)

            for j, cur_model in enumerate(self.models):
                model = deepcopy(cur_model)
                model.fit(train_dict)
                model_train_res = model.predict(train_dict)
                model_test_res = model.predict(test_dict)
                self._write_fold_info_to_dict(res_dict, f'model_{j}', metrics,
                                              train_dict, model_train_res,
                                              test_dict, model_test_res)

        return res_dict

from .base import BaseComparator, BaseForeignModel
from .metric import BaseMetric
from bosk.executor.base import BaseExecutor
from bosk.data import BaseData
from bosk.pipeline.base import BasePipeline
from bosk.stages import Stage
from bosk.executor.topological import TopologicalExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.utility import timer_wrap
from collections import defaultdict
from copy import deepcopy
import numpy as np
from typing import List, Dict, Optional
from sklearn.model_selection import BaseCrossValidator
from pandas import DataFrame
import logging
import warnings


class CVComparator(BaseComparator):
    """Comparator that uses a cross-validation
    strategy of checking model's performance. You can create
    your own iterator (based on sklearn's `BaseCrossValidator` class)
    that will define indexes, taken in each fold,
    or you can use predefined iterators from the `sklearn`.
    """

    def _write_fold_info_to_dict(self, df_dict, metrics,
                                 train_data_dict, train_pred_dict,
                                 test_data_dict, test_pred_dict) -> None:
        for i, metric in enumerate(metrics):
            df_dict[self._metrics_names[i]].append(
                metric.get_score(train_data_dict, train_pred_dict))
            df_dict[self._metrics_names[i]].append(
                metric.get_score(test_data_dict, test_pred_dict))

    def _get_pers_inp_dict(self, pipeline, common_output, input_dict) -> Dict[str, BaseData]:
        personal_dict = dict()
        for key in pipeline.inputs.keys():
            if key in input_dict:
                personal_dict[key] = input_dict[key]
            elif key in common_output:
                personal_dict[key] = common_output[key]
            else:
                raise RuntimeError(
                    f"Unable to find '{key}' key neither in the data nor in the common part")
        return personal_dict

    def _get_metrics_names(self, metrics: List[BaseMetric]) -> List[str]:
        names = []
        for i, metric in enumerate(metrics):
            name = metric.name if metric.name is not None else f'metric_{i}'
            names.append(name)
        return names

    def _write_preamble(self, df_dict, model_name, fold_num) -> None:
        df_dict['model name'] += [model_name] * 2
        df_dict['fold #'] += [fold_num] * 2
        df_dict['train/test'] += ['train', 'test']

    def __init__(self, pipelines: List[BasePipeline], common_part: BasePipeline,
                 foreign_models: List[BaseForeignModel], cv_strat: BaseCrossValidator,
                 exec_cls: BaseExecutor = TopologicalExecutor, suppress_exec_warn: bool = True, 
                 random_state: Optional[int] = None) -> None:
        super().__init__(pipelines, common_part, foreign_models, random_state)
        cv_strat.random_state = random_state
        self.cv_strat = cv_strat
        self.exec_cls = exec_cls
        self.warn_context = 'ignore' if suppress_exec_warn else 'default'

    def get_score(self, data: Dict[str, BaseData], metrics: List[BaseMetric]) -> DataFrame:
        n = None
        for value in data.values():
            if n is None:
                n = value.data.shape[0]
            else:
                assert value.data.shape[0] == n, "All inputs must have the same number of samples"

        self._metrics_names = self._get_metrics_names(metrics)
        idx = np.arange(n)
        dataframe_dict = defaultdict(list)
        for i, (train_idx, test_idx) in enumerate(self.cv_strat.split(idx)):
            logging.info('Processing fold #%i', i)
            
            # getting fold subsets of data
            train_data_dict, test_data_dict = dict(), dict()
            for key, val in data.items():
                train_data_dict[key] = val.__class__(val.data[train_idx])
                test_data_dict[key] = val.__class__(val.data[test_idx])

            # processing the common part
            if self.common_pipeline is None:
                common_train_res = dict()
                common_test_res = common_train_res
                train_common_part_time = 0
                test_common_part_time = 0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(self.warn_context)
                    train_exec = self.exec_cls(self.common_pipeline,
                                            HandlingDescriptor.from_classes(Stage.FIT))
                    common_train_res, train_common_part_time = timer_wrap(train_exec)(train_data_dict)

                    test_exec = self.exec_cls(self.common_pipeline,
                                            HandlingDescriptor.from_classes(Stage.TRANSFORM))
                    common_test_res, test_common_part_time = timer_wrap(test_exec)(test_data_dict)

            for j, cur_pipeline in enumerate(self.optim_pipelines):
                pip_name = f'deep forest {j}'
                self._write_preamble(dataframe_dict, pip_name, i)

                pipeline = deepcopy(cur_pipeline)
                cur_train_dict = self._get_pers_inp_dict(pipeline, common_train_res, train_data_dict)
                cur_test_dict = self._get_pers_inp_dict(pipeline, common_test_res, test_data_dict)
                with warnings.catch_warnings():
                    warnings.simplefilter(self.warn_context)
                    pip_tr_exec = self.exec_cls(pipeline, HandlingDescriptor.from_classes(Stage.FIT))
                    pip_train_res, train_time = timer_wrap(pip_tr_exec)(cur_train_dict)
                    dataframe_dict['time'].append(train_common_part_time + train_time)

                    pip_test_exec = self.exec_cls(pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM))
                    pip_test_res, test_time = timer_wrap(pip_test_exec)(cur_test_dict)
                    dataframe_dict['time'].append(test_common_part_time + test_time)

                self._write_fold_info_to_dict(dataframe_dict, metrics,
                                              train_data_dict, pip_train_res,
                                              test_data_dict, pip_test_res)

            for j, cur_model in enumerate(self.models):
                model_name = f'model {j}'
                self._write_preamble(dataframe_dict, model_name, i)
                model = deepcopy(cur_model)
                model.fit(train_data_dict)
                model_train_res, model_train_time = timer_wrap(model.predict)(train_data_dict)
                model_test_res, model_test_time = timer_wrap(model.predict)(test_data_dict)
                dataframe_dict['time'].append(model_train_time)
                dataframe_dict['time'].append(model_test_time)
                self._write_fold_info_to_dict(dataframe_dict, metrics,
                                              train_data_dict, model_train_res,
                                              test_data_dict, model_test_res)

        del self._metrics_names  # it's a helper field only for this func call
        return DataFrame(dataframe_dict)

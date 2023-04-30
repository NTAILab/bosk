from .base import BaseComparator, BaseForeignModel
from .metric import BaseMetric
from bosk.data import BaseData
from bosk.block.base import BaseBlock
from bosk.executor.base import BaseExecutor, DefaultBlockExecutor
from bosk.executor.timer import TimerBlockExecutor
from bosk.pipeline.base import BasePipeline
from bosk.stages import Stage
from bosk.executor.topological import TopologicalExecutor
from bosk.utility import timer_wrap
from collections import defaultdict
from copy import deepcopy
import numpy as np
from typing import List, Dict, Literal, Optional, Tuple, Type
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

    def __init__(self, pipelines: Optional[BasePipeline | List[BasePipeline]],
                 foreign_models: Optional[BaseForeignModel | List[BaseForeignModel]],
                 cv_strat: BaseCrossValidator, exec_cls: Type[BaseExecutor] = TopologicalExecutor,
                 exec_kw=None, get_blocks_times: bool = False, suppress_exec_warn: bool = True,
                 f_optimize_pipelines: bool = True, random_state: Optional[int] = None) -> None:
        super().__init__(pipelines, foreign_models, f_optimize_pipelines, random_state)
        cv_strat.random_state = random_state
        self.cv_strat = cv_strat
        self.exec_cls = exec_cls
        if exec_kw is None:
            self.exec_kw = dict()
        else:
            forbidden_exec_args = ['pipeline', 'stage', 'block_executor']
            if any([key in exec_kw for key in forbidden_exec_args]):
                raise RuntimeError(
                    f"You mustn't specify following args for executor: {forbidden_exec_args}")
            self.exec_kw = exec_kw
        self.measure_blk_time = get_blocks_times
        self.block_hlr_cls = TimerBlockExecutor if get_blocks_times else DefaultBlockExecutor
        self.warn_context: Literal['ignore'] | Literal['default'] = 'ignore' if suppress_exec_warn else 'default'

    def _write_metrics_info_to_dict(self, df_dict, metrics,
                                    train_data_dict, train_pred_dict,
                                    test_data_dict, test_pred_dict,
                                    metrics_names) -> None:
        for i, metric in enumerate(metrics):
            df_dict[metrics_names[i]].append(
                metric.get_score(train_data_dict, train_pred_dict))
            df_dict[metrics_names[i]].append(
                metric.get_score(test_data_dict, test_pred_dict))

    # copy isomorphism is returned only for blocks' time measuring case
    def _get_copy_pipeline(self, pip_num: int) -> Tuple[BasePipeline, Dict[BaseBlock, BaseBlock] | None]:
        orig_pip = self._optim_pipelines[pip_num]
        pip_copy = deepcopy(orig_pip)
        if not self.measure_blk_time:
            return pip_copy, None
        extra_blocks_num = len(self._new_blocks_list[pip_num])
        copy_iso = dict()  # copy_pip-> orig_pip
        for i in range(len(orig_pip.nodes) - extra_blocks_num):
            copy_iso[pip_copy.nodes[i]] = orig_pip.nodes[i]
        return pip_copy, copy_iso

    def _get_times_dict(self, common_times, pip_times,
                        common_part_iso, copy_iso) -> Dict[BaseBlock, float]:
        res = dict()
        for block, time in pip_times.items():
            orig_block = copy_iso.get(block, None)
            if orig_block is not None:
                res[orig_block] = time
        for block, time in common_times.items():
            res[common_part_iso[block]] = time
        return res

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

    def _write_preamble(self, df_dict: defaultdict[str, List], model_name, fold_num) -> None:
        df_dict['model name'] += [model_name] * 2
        df_dict['fold #'] += [fold_num] * 2
        df_dict['train/test'] += ['train', 'test']

    # columns: model name | fold # | train/test | time | blocks time | metric name 1 | ... | metric name n
    def get_score(self, data: Dict[str, BaseData], metrics: List[BaseMetric]) -> DataFrame:
        n = None
        for value in data.values():
            if n is None:
                n = value.data.shape[0]
            else:
                assert value.data.shape[0] == n, "All inputs must have the same number of samples"
        assert n is not None

        metrics_names = self._get_metrics_names(metrics)
        idx = np.arange(n)
        dataframe_dict: defaultdict[str, List] = defaultdict(list)

        for i, (train_idx, test_idx) in enumerate(self.cv_strat.split(idx)):
            logging.info('Processing fold #%i', i)

            # getting fold subsets of data
            train_data_dict, test_data_dict = dict(), dict()
            for key, val in data.items():
                train_data_dict[key] = val.__class__(val.data[train_idx])
                test_data_dict[key] = val.__class__(val.data[test_idx])

            # processing the common part
            if self._common_pipeline is None:
                common_train_res: Dict[str, BaseData] = dict()
                common_test_res: Dict[str, BaseData] = dict()
                train_common_part_time = 0.0
                test_common_part_time = 0.0
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(self.warn_context)
                    block_exec = self.block_hlr_cls()
                    train_exec = self.exec_cls(self._common_pipeline,
                                               Stage.FIT, block_executor=block_exec, **self.exec_kw)
                    common_train_res, train_common_part_time = timer_wrap(
                        train_exec)(train_data_dict)
                    if self.measure_blk_time:
                        assert isinstance(block_exec, TimerBlockExecutor)
                        common_block_train_times = block_exec.blocks_time
                    block_exec = self.block_hlr_cls()
                    test_exec = self.exec_cls(
                        self._common_pipeline, Stage.TRANSFORM, block_executor=block_exec, **self.exec_kw)
                    common_test_res, test_common_part_time = timer_wrap(test_exec)(test_data_dict)
                    if self.measure_blk_time:
                        assert isinstance(block_exec, TimerBlockExecutor)
                        common_block_test_times = block_exec.blocks_time

            for j in range(len(self._optim_pipelines)):
                pip_name = f'deep forest {j}'
                self._write_preamble(dataframe_dict, pip_name, i)

                pipeline, copy_iso = self._get_copy_pipeline(j)
                cur_train_dict = self._get_pers_inp_dict(
                    pipeline, common_train_res, train_data_dict)
                cur_test_dict = self._get_pers_inp_dict(pipeline, common_test_res, test_data_dict)
                with warnings.catch_warnings():
                    warnings.simplefilter(self.warn_context)
                    block_exec = self.block_hlr_cls()
                    pip_tr_exec = self.exec_cls(
                        pipeline, Stage.FIT, block_executor=block_exec, **self.exec_kw)
                    pip_train_res, train_time = timer_wrap(pip_tr_exec)(cur_train_dict)
                    dataframe_dict['time'].append(train_common_part_time + train_time)
                    if self.measure_blk_time:
                        assert isinstance(block_exec, TimerBlockExecutor)
                        pip_block_train_times = block_exec.blocks_time
                    block_exec = self.block_hlr_cls()
                    pip_test_exec = self.exec_cls(
                        pipeline, Stage.TRANSFORM, block_executor=block_exec, **self.exec_kw)
                    pip_test_res, test_time = timer_wrap(pip_test_exec)(cur_test_dict)
                    dataframe_dict['time'].append(test_common_part_time + test_time)
                    if self.measure_blk_time:
                        assert isinstance(block_exec, TimerBlockExecutor)
                        pip_block_test_times = block_exec.blocks_time
                        orig_pip_train_times = self._get_times_dict(
                            common_block_train_times, pip_block_train_times,
                            self._block_iso_list[j], copy_iso
                        )
                        dataframe_dict['blocks time'].append(orig_pip_train_times)
                        orig_pip_test_times = self._get_times_dict(
                            common_block_test_times, pip_block_test_times,
                            self._block_iso_list[j], copy_iso
                        )
                        dataframe_dict['blocks time'].append(orig_pip_test_times)

                self._write_metrics_info_to_dict(dataframe_dict, metrics,
                                                 train_data_dict, pip_train_res,
                                                 test_data_dict, pip_test_res,
                                                 metrics_names)

            for j, cur_model in enumerate(self.models):
                model_name = f'model {j}'
                self._write_preamble(dataframe_dict, model_name, i)
                model = deepcopy(cur_model)
                model.fit(train_data_dict)
                model_train_res, model_train_time = timer_wrap(model.predict)(train_data_dict)
                model_test_res, model_test_time = timer_wrap(model.predict)(test_data_dict)
                dataframe_dict['time'].append(model_train_time)
                dataframe_dict['time'].append(model_test_time)
                self._write_metrics_info_to_dict(dataframe_dict, metrics,
                                                 train_data_dict, model_train_res,
                                                 test_data_dict, model_test_res,
                                                 metrics_names)
                if self.measure_blk_time:
                    dataframe_dict['blocks time'] += [None] * 2

        return DataFrame(dataframe_dict)

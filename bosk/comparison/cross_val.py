from .base import BaseComparator
from bosk.executor.base import BaseExecutor
from bosk.data import BaseData
from bosk.pipeline.base import BasePipeline
import numpy as np
from bosk.stages import Stage
from typing import List, Callable, Dict
from bosk.executor.topological import TopologicalExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.painter.topological import TopologicalPainter
from sklearn.model_selection import BaseCrossValidator
import logging


class CVComparator(BaseComparator):
    def __init__(self, pipelines: List[BasePipeline], common_part: BasePipeline,
                 cv_strat: BaseCrossValidator, exec_cls: BaseExecutor = TopologicalExecutor) -> None:
        super().__init__(pipelines, common_part)
        self.cv_strat = cv_strat
        self.exec_cls = exec_cls

    def get_score(self, data: Dict[str, BaseData], metrics: List[Callable]):
        n = None
        for value in data.values():
            if n is None:
                n = value.data.shape[0]
            else:
                assert value.data.shape[0] == n, "All inputs must have the same number of samples"
        idx = np.arange(n)
        metrics_train_hist = []
        metrics_test_hist = []
        for i in range(len(self.optim_pipelines)):
            metrics_train_hist.append([[] for _ in range(len(metrics))])
            metrics_test_hist.append([[] for _ in range(len(metrics))])
        for i, (train_idx, test_idx) in enumerate(self.cv_strat.split(idx)):
            logging.info('Fold %i', i)

            train_dict = dict()
            test_dict = dict()
            for key, val in data.items():
                train_dict[key] = val.__class__(val.data[train_idx])
                test_dict[key] = val.__class__(val.data[test_idx])

            train_exec = TopologicalExecutor(self.common_pipeline,
                                             HandlingDescriptor.from_classes(Stage.FIT))
            common_train_res = train_exec(train_dict)
            test_exec = TopologicalExecutor(self.common_pipeline,
                                            HandlingDescriptor.from_classes(Stage.TRANSFORM))
            common_test_res = test_exec(test_dict)

            for j, pipeline in enumerate(self.optim_pipelines):
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

                pip_tr_exec = TopologicalExecutor(pipeline,
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

                pip_test_exec = TopologicalExecutor(pipeline,
                                                    HandlingDescriptor.from_classes(Stage.TRANSFORM))
                pip_test_res = pip_test_exec(cur_test_dict)

                # todo: debug feature, remove the painting after
                if i == 0:
                    TopologicalPainter().from_executor(train_exec).render('common_pipeline_fit.png')
                    TopologicalPainter().from_executor(test_exec).render('common_pipeline_tf.png')
                    TopologicalPainter().from_executor(pip_tr_exec).render(f'optim_pipeline_{j}_fit.png')
                    TopologicalPainter().from_executor(pip_test_exec).render(f'optim_pipeline_{j}_tf.png')

                for k, metric in enumerate(metrics):
                    metric_train_res = metric(train_dict, pip_train_res)
                    metrics_train_hist[j][k].append(metric_train_res)
                    logging.info(f'\tTrain fold res (model {j} metric {k}): %f', metric_train_res)
                    metric_test_res = metric(test_dict, pip_test_res)
                    metrics_test_hist[j][k].append(metric_test_res)
                    logging.info(f'\tTest fold res (model {j} metric {k}): %f', metric_test_res)

        logging.info(f'Average results:')
        for i in range(len(self.optim_pipelines)):
            logging.info(f'Model {i}:')
            for j in range(len(metrics)):
                logging.info(f'\tMetric {j} train: %f', np.mean(metrics_train_hist[i][j]))
                logging.info(f'\tMetric {j} test: %f', np.mean(metrics_test_hist[i][j]))

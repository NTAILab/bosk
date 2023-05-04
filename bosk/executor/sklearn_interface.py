"""Scikit-learn wrappers for BOSK pipelines.

"""
import numpy as np
from typing import List, Mapping, Optional, Type
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.multiclass import check_classification_targets
from .base import BaseExecutor, BasePipeline, Stage
from ..data import CPUData
from .topological import TopologicalExecutor


__all__ = [
    "BaseBoskPipelineWrapper",
    "BoskPipelineClassifier",
    "BoskPipelineRegressor",
    "BoskPipelineTransformer",
]


DEFAULT_EXECUTOR = TopologicalExecutor


class BaseBoskPipelineWrapper(BaseEstimator):
    """Base BOSK Pipeline Wrapper is a base pipeline execution wrapper for
    matching scikit-learn interface.

    The wrapper executes the given pipeline using two separate executors:
    for FIT and TRANSFORM stages.

    """
    def __init__(self, pipeline: BasePipeline,
                 inputs_map: Optional[Mapping[str, str]] = None,
                 outputs_map: Optional[Mapping[str, str]] = None,
                 executor_cls: Type[BaseExecutor] = DEFAULT_EXECUTOR):
        """Initialize the BOSK Pipeline Wrapper.

        Args:
            pipeline: The BOSK pipeline.
            inputs_map: Mapping from the fixed set `{'X', 'y', 'sample_weight'}`
                        to the names of corresponding inputs of the pipeline.
                        If some inputs are not used, input_map should not contain them.
                        `None` is equivalent to the mapping `{'X': 'X', ...}`,
                        where each key corresponds to the pipeline input slot.
            outputs_map: Mapping from the fixed set {'pred', 'proba'}
                         to the names of corresponding outputs of the pipeline:
                         'pred' means predictions and 'proba' means class probabilities.
                        `None` is equivalent to the mapping `{'pred': 'pred', ...}`,
                        where each key corresponds to the pipeline output slot.
            executor_cls: BOSK Pipeline executor class.
                          The default is `TopologicalExecutor`.


        """
        self.pipeline = pipeline
        self.executor_cls = executor_cls
        self.inputs_map = inputs_map
        self.outputs_map = outputs_map
        self._prepare_executors()

    def __map_vars(self, keys: List[str], mapping: Mapping[str, str]) -> List[str]:
        return [
            mapping[k]
            for k in keys
            if k in mapping
        ]

    def _prepare_executors(self):
        FIT_INPUT_ARGS = ['X', 'y', 'sample_weight']
        OUTPUT_VARIANTS = ['pred', 'proba']
        if self.inputs_map is None:
            self.inputs_map = {
                k: k
                for k in FIT_INPUT_ARGS
                if k in self.pipeline.inputs
            }
        if self.outputs_map is None:
            self.outputs_map = {
                k: k
                for k in OUTPUT_VARIANTS
                if k in self.pipeline.outputs
            }
        self.fit_executor_ = self.executor_cls(
            self.pipeline,
            stage=Stage.FIT,
            inputs=self.__map_vars(FIT_INPUT_ARGS, self.inputs_map),
            outputs=self.__map_vars(OUTPUT_VARIANTS, self.outputs_map),
        )
        self.transform_executor_ = self.executor_cls(
            self.pipeline,
            stage=Stage.TRANSFORM,
            inputs=self.__map_vars(['X'], self.inputs_map),
            outputs=self.__map_vars(OUTPUT_VARIANTS, self.outputs_map),
        )

    def fit(self, X, y, sample_weight=None, **kwargs):
        inputs = dict(X=X, y=y, sample_weight=sample_weight, **kwargs)
        args = {
            self.inputs_map[k]: CPUData(v)
            for k, v in inputs.items()
            if k in self.inputs_map
        }
        self.fit_executor_(args)
        return self

    def _predict_all(self, X, **kwargs):
        inputs = dict(X=X, **kwargs)
        args = {
            self.inputs_map[k]: CPUData(v)
            for k, v in inputs.items()
            if k in self.inputs_map
        }
        result = self.transform_executor_(args)
        return result

    def _extract(self, key, result):
        """Extract the output value by key from execution result.
        """
        mapped_key = self.outputs_map[key]
        if mapped_key not in result:
            raise RuntimeError(
                f'Key {mapped_key!r} (originally {key!r}) does not present '
                f'in the execution result ({list(result.keys())!r})'
            )
        return result[mapped_key].to_cpu().data


class BoskPipelineClassifier(ClassifierMixin, BaseBoskPipelineWrapper):
    """Classifier based on BOSK pipeline.
    """
    def __init__(self, pipeline: BasePipeline,
                 inputs_map: Optional[Mapping[str, str]] = None,
                 outputs_map: Optional[Mapping[str, str]] = None,
                 executor_cls: Type[BaseExecutor] = DEFAULT_EXECUTOR):
        super().__init__(
            pipeline=pipeline,
            inputs_map=inputs_map,
            outputs_map=outputs_map,
            executor_cls=executor_cls,
        )

    def _classifier_init(self, y):
        check_classification_targets(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        y = y_encoded
        return y

    def fit(self, X, y, sample_weight=None, **kwargs):
        y = self._classifier_init(y)
        return super().fit(X, y, sample_weight, **kwargs)

    def predict_proba(self, X, **kwargs):
        assert 'proba' in self.outputs_map
        return self._extract('proba', self._predict_all(X, **kwargs))

    def predict(self, X, **kwargs):
        assert 'pred' in self.outputs_map or 'proba' in self.outputs_map, \
            f'At least "pred" or "proba" keys must present in outputs_map ({self.outputs_map!r})'
        result = self._predict_all(X, **kwargs)
        if 'pred' in self.outputs_map:
            return self._extract('pred', result)
        proba = self._extract('proba', result)
        return self.classes_[np.argmax(proba, axis=1)]


class BoskPipelineRegressor(RegressorMixin, BaseBoskPipelineWrapper):
    """Regressor based on BOSK pipeline.
    """
    def __init__(self, pipeline: BasePipeline,
                 inputs_map: Optional[Mapping[str, str]] = None,
                 outputs_map: Optional[Mapping[str, str]] = None,
                 executor_cls: Type[BaseExecutor] = DEFAULT_EXECUTOR):
        super().__init__(
            pipeline=pipeline,
            inputs_map=inputs_map,
            outputs_map=outputs_map,
            executor_cls=executor_cls,
        )

    def predict(self, X, **kwargs):
        assert 'pred' in self.outputs_map
        return self._extract('pred', self._predict_all(X, **kwargs))


class BoskPipelineTransformer(TransformerMixin, BaseBoskPipelineWrapper):
    """Transformer based on BOSK pipeline.
    """
    def __init__(self, pipeline: BasePipeline,
                 inputs_map: Optional[Mapping[str, str]] = None,
                 outputs_map: Optional[Mapping[str, str]] = None,
                 executor_cls: Type[BaseExecutor] = DEFAULT_EXECUTOR):
        super().__init__(
            pipeline=pipeline,
            inputs_map=inputs_map,
            outputs_map=outputs_map,
            executor_cls=executor_cls,
        )

    def fit_transform(self, X, y, sample_weight=None, **kwargs):
        inputs = dict(X=X, y=y, sample_weight=sample_weight, **kwargs)
        args = {
            self.inputs_map[k]: CPUData(v)
            for k, v in inputs.items()
            if k in self.inputs_map
        }
        result = self.fit_executor_(args)
        return self._extract('pred', result)

    def transform(self, X, **kwargs):
        assert 'pred' in self.outputs_map
        return self._extract('pred', self._predict_all(X, **kwargs))

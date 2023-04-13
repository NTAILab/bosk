from __future__ import annotations

from functools import partial
from typing import Any, Optional, Type

import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.tree_util import register_pytree_node_class

# from .classifier import DecisionTreeClassifier
# from .core import DecisionTree
# from .regressor import DecisionTreeRegressor


from collections import defaultdict

from functools import partial
from typing import Callable, Tuple, Dict, List

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.tree_util import register_pytree_node_class


from ....base import BaseBlock, TransformOutputData, BlockInputData
from ....meta import BlockMeta, BlockExecutionProperties
from ....slot import InputSlotMeta, OutputSlotMeta
from .....stages import Stages
from .....data import GPUData
from .....utility import get_random_generator, get_rand_int
from .classification_models_jax_util import DecisionTreeClassifier


@register_pytree_node_class
class RandomForestClassifierBlock(BaseBlock):
    """Base class for random forests."""

    meta = BlockMeta(
        inputs=[
            InputSlotMeta(
                name='X',
                stages=Stages(transform=True),
            ),
            InputSlotMeta(
                name='y',
                stages=Stages(transform=False, transform_on_fit=False),
            )
        ],
        outputs=[
            OutputSlotMeta(
                name='probas',
            )
        ],
        execution_props=BlockExecutionProperties(gpu=True),
    )

    def __init__(
        self,
        n_classes: int,
        n_estimators: int = 100,
        min_samples: int = 2,
        max_depth: int = 4,
        max_splits: int = 25,
        predictors: Optional[DecisionTreeClassifier] = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.base_class = DecisionTreeClassifier
        self.base_model = DecisionTreeClassifier(n_classes=n_classes, **kwargs)
        self.n_estimators = n_estimators
        self.predictors: Optional[DecisionTreeClassifier] = None
        self._n_classes = n_classes
        self.aux_data = {
            "n_estimators": n_estimators,
        }
        self.aux_data.update(**kwargs)

    def tree_flatten(self):
        children = [self.predictors]
        return (children, self.aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (predictors,) = children
        return cls(**aux_data, n_classes=2, predictors=predictors)

    def fit(self, inputs: BlockInputData):
        X = inputs["X"].data
        y = inputs["y"].data
        return self.fit_jit(X, y)

    # @jit
    def fit_jit(self, X: jnp.ndarray, y: jnp.ndarray):
        print(f"fit {id(self)}")
        n_samples = X.shape[0]
        key = jax.random.PRNGKey(seed=0)
        idx = jax.random.randint(
            key,
            shape=(self.n_estimators, n_samples),
            minval=0,
            maxval=n_samples,
        )
        # Bootstrap dataset by creating a mask, weighting each sample with a
        # positive integer, the mask's sum being equal to n_samples
        mask = vmap(partial(jnp.bincount, length=n_samples))(idx)
        self.predictors = vmap(
            self.base_class.fit, in_axes=[None, None, None, 0]
        )(self.base_model, X, y, mask)
        return self

    def transform(self, inputs: BlockInputData) -> TransformOutputData:
        print(f"transform {id(self)}")
        res = self.predict(inputs["X"].data)
        return {"probas": GPUData(res)}

    # @jit
    def predict(self, X: jnp.ndarray) -> jnp.ndarray:
        if self.predictors is None:
            raise ValueError("The model is not fitted.")

        preds = vmap(self.base_class.predict, in_axes=[0, None])(
            self.predictors, X
        )
        return jnp.mean(preds, axis=0)

    def score(self, X: jnp.ndarray, y: jnp.ndarray) -> float:
        preds = self.predict(X)
        return self.base_model.score_fn(preds, y)


# @register_pytree_node_class
# class RandomForestClassifier(RandomForest):
#     def __init__(
#         self,
#         n_classes: int,
#         n_estimators: int = 100,
#         min_samples: int = 2,
#         max_depth: int = 4,
#         max_splits: int = 25,
#         predictors: Optional[DecisionTreeClassifier] = None,
#     ):
#         super().__init__(
#             n_classes=n_classes,
#             n_estimators=n_estimators,
#             min_samples=min_samples,
#             max_depth=max_depth,
#             max_splits=max_splits,
#             base_class=DecisionTreeClassifier,
#             predictors=predictors,
#         )

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from ....auto import auto_block
from ....meta import BlockExecutionProperties
from .....data import CPUData, np, jnp


if jnp != np:
    from .ferns import RandomFernsBlock
    from .mgs_ferns import MGSRandomFernsBlock
    from .jax import RFCGBlock, ETCGBlock



__all__ = [
    "RFCBlock",
    "RFCGBlock",
    "ETCGBlock",
    "ETCBlock",
    "CatBoostClassifierBlock",
    "XGBClassifierBlock",
    "RandomFernsBlock",
    "MGSRandomFernsBlock",
]


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class RFCBlock(RandomForestClassifier):
    """Random Forest Classifier.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Input features.
        - y: Ground truth labels.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Input features.

    Output slots
    ------------

        - output: Predicted probabilities.

    """
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ETCBlock(ExtraTreesClassifier):
    """Extremely Randomized Trees Classifier.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Input features.
        - y: Ground truth labels.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Input features.

    Output slots
    ------------

        - output: Predicted probabilities.

    """
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties())
class CatBoostClassifierBlock(CatBoostClassifier):
    """CatBoost Classifier.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Input features.
        - y: Ground truth labels.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Input features.

    Output slots
    ------------

        - output: Predicted probabilities.

    """
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBClassifierBlock(XGBClassifier):
    """XGBoost Classifier.

    Input slots
    -----------

    Fit inputs
    ~~~~~~~~~~

        - X: Input features.
        - y: Ground truth labels.

    Transform inputs
    ~~~~~~~~~~~~~~~~

        - X: Input features.

    Output slots
    ------------

        - output: Predicted probabilities.

    """
    def fit(self, X, y):
        super().fit(X, y)

    def transform(self, X):
        return CPUData(self.predict_proba(X))


try:
    from lightgbm import LGBMClassifier

    @auto_block(execution_props=BlockExecutionProperties())
    class LGBMClassifierBlock(LGBMClassifier):
        """LightGBM Classifier.

        Input slots
        -----------

        Fit inputs
        ~~~~~~~~~~

            - X: Input features.
            - y: Ground truth labels.

        Transform inputs
        ~~~~~~~~~~~~~~~~

            - X: Input features.

        Output slots
        ------------

            - output: Predicted probabilities.

        """
        def fit(self, X, y):
            super().fit(X, y)

        def transform(self, X):
            return CPUData(self.predict_proba(X))

except ModuleNotFoundError:
    pass

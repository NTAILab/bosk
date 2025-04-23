from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from ....auto import auto_block
from ....meta import BlockExecutionProperties
from .....data import CPUData, np, jnp


if jnp != np:
    from .ferns import RandomFerns, RandomFernsBlock
    from .mgs_ferns import MGSRandomFerns, MGSRandomFernsBlock
    from .jax import RFCG, ETCG, RFCGBlock, ETCGBlock



__all__ = [
    "RFC",
    "RFCG",
    "ETC",
    "ETCG",
    "CatBoostClassifier",
    "XGBClassifier",
    "RandomFerns",
    "MGSRandomFerns",
    # for backward compatibility:
    "RFCBlock",
    "RFCGBlock",
    "ETCGBlock",
    "ETCBlock",
    "CatBoostClassifierBlock",
    "XGBClassifierBlock",
    "RandomFernsBlock",
    "MGSRandomFernsBlock",
]


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True), fit_argnames={'X', 'y', 'sample_weight'})
class RFC(RandomForestClassifier):
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


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True), fit_argnames={'X', 'y', 'sample_weight'})
class ETC(ExtraTreesClassifier):
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


try:
    from catboost import CatBoostClassifier as _CatBoostClassifier

    @auto_block(execution_props=BlockExecutionProperties())
    class CatBoostClassifier(_CatBoostClassifier):
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

    CatBoostClassifierBlock = CatBoostClassifier

except ModuleNotFoundError:
    pass


try:
    from xgboost import XGBClassifier as _XGBClassifier

    @auto_block(execution_props=BlockExecutionProperties())
    class XGBClassifier(_XGBClassifier):
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

    XGBClassifierBlock = XGBClassifier

except ModuleNotFoundError:
    pass


RFCBlock = RFC
ETCBlock = ETC


try:
    from lightgbm import LGBMClassifier as _LGBMClassifier

    @auto_block(execution_props=BlockExecutionProperties())
    class LGBMClassifier(_LGBMClassifier):
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

    LGBMClassifierBlock = LGBMClassifier

except ModuleNotFoundError:
    pass


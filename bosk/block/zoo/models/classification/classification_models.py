from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from ....auto import auto_block
from ....meta import BlockExecutionProperties
from .....data import CPUData


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class RFCBlock(RandomForestClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ETCBlock(ExtraTreesClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties())
class CatBoostClassifierBlock(CatBoostClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBClassifierBlock(XGBClassifier):
    def fit(self, X, y):
        super().fit(X, y)

    def transform(self, X):
        return CPUData(self.predict_proba(X))

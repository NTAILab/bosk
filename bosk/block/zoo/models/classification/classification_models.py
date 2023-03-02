from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from bosk.block import auto_block
from bosk.block.meta import BlockExecutionProperties
from bosk.data import CPUData


@auto_block(execution_props=BlockExecutionProperties(cpu=True, gpu=False))
class RFCBlock(RandomForestClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(cpu=True, gpu=False))
class ETCBlock(ExtraTreesClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(cpu=True, gpu=False))
class CatBoostClassifierBlock(CatBoostClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(cpu=True, gpu=False))
class XGBClassifierBlock(XGBClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))

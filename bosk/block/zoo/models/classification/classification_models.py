from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from bosk.block import auto_block
from bosk.block.meta import BlockExecutionProperties
from bosk.data import CPUData


@auto_block(execution_props=BlockExecutionProperties())
class RFCBlock(RandomForestClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties())
class ETCBlock(ExtraTreesClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties(), random_state_field='random_seed_')
class CatBoostClassifierBlock(CatBoostClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBClassifierBlock(XGBClassifier):
    def transform(self, X):
        return CPUData(self.predict_proba(X))

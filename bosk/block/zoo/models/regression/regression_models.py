from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from bosk.block import auto_block
from bosk.block.meta import BlockExecutionProperties
from bosk.data import CPUData


@auto_block(execution_props=BlockExecutionProperties())
class RFRBlock(RandomForestRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class ETRBlock(ExtraTreesRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class CatBoostRegressorBlock(CatBoostRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBCRegressorBlock(XGBRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))

from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from ....auto import auto_block
from ....meta import BlockExecutionProperties
from .....data import CPUData


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class RFRBlock(RandomForestRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ETRBlock(ExtraTreesRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties(), random_state_field='random_seed_')
class CatBoostRegressorBlock(CatBoostRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBCRegressorBlock(XGBRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))

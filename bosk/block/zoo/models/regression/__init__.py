from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from ....auto import auto_block
from ....meta import BlockExecutionProperties
from .....data import CPUData


__all__ = [
    "RFRBlock",
    "ETRBlock",
    "CatBoostRegressorBlock",
    "XGBCRegressorBlock",
]


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class RFRBlock(RandomForestRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ETRBlock(ExtraTreesRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class CatBoostRegressorBlock(CatBoostRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBCRegressorBlock(XGBRegressor):
    def fit(self, X, y):
        super().fit(X, y)

    def transform(self, X):
        return CPUData(self.predict(X))


try:
    from lightgbm import LGBMRegressor

    @auto_block(execution_props=BlockExecutionProperties())
    class LGBMRegressorBlock(LGBMRegressor):
        def fit(self, X, y):
            super().fit(X, y)

        def transform(self, X):
            return CPUData(self.predict(X))

except ModuleNotFoundError:
    pass

from catboost import CatBoostRegressor as _CatBoostRegressor
from xgboost import XGBRegressor as _XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from ....auto import auto_block
from ....meta import BlockExecutionProperties
from .....data import CPUData


__all__ = [
    "RFR",
    "ETR",
    "CatBoostRegressor",
    "XGBCRegressor",
    # for backward compatibility:
    "RFRBlock",
    "ETRBlock",
    "CatBoostRegressorBlock",
    "XGBCRegressorBlock",
]


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class RFR(RandomForestRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True))
class ETR(ExtraTreesRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class CatBoostRegressor(_CatBoostRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties())
class XGBCRegressor(_XGBRegressor):
    def fit(self, X, y):
        super().fit(X, y)

    def transform(self, X):
        return CPUData(self.predict(X))


RFRBlock = RFR
ETRBlock = ETR
CatBoostRegressorBlock = CatBoostRegressor
XGBCRegressorBlock = XGBCRegressor


try:
    from lightgbm import LGBMRegressor as _LGBMRegressor

    @auto_block(execution_props=BlockExecutionProperties())
    class LGBMRegressor(_LGBMRegressor):
        def fit(self, X, y):
            super().fit(X, y)

        def transform(self, X):
            return CPUData(self.predict(X))


    LGBMRegressorBlock = LGBMRegressor

except ModuleNotFoundError:
    pass

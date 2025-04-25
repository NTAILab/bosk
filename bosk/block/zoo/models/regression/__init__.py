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


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True), fit_argnames={'X', 'y', 'sample_weight'})
class RFR(RandomForestRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


@auto_block(execution_props=BlockExecutionProperties(threadsafe=True), fit_argnames={'X', 'y', 'sample_weight'})
class ETR(ExtraTreesRegressor):
    def transform(self, X):
        return CPUData(self.predict(X))


try:
    from catboost import CatBoostRegressor as _CatBoostRegressor

    @auto_block(execution_props=BlockExecutionProperties())
    class CatBoostRegressor(_CatBoostRegressor):
        def transform(self, X):
            return CPUData(self.predict(X))


    CatBoostRegressorBlock = CatBoostRegressor

except ModuleNotFoundError:
    pass


try:
    from xgboost import XGBRegressor as _XGBRegressor

    @auto_block(execution_props=BlockExecutionProperties())
    class XGBCRegressor(_XGBRegressor):
        def fit(self, X, y):
            super().fit(X, y)

        def transform(self, X):
            return CPUData(self.predict(X))


    XGBCRegressorBlock = XGBCRegressor

except ModuleNotFoundError:
    pass


RFRBlock = RFR
ETRBlock = ETR


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

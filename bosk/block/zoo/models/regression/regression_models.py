from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from bosk.block import auto_block


@auto_block
class RFRBlock(RandomForestRegressor):
    def transform(self, X):
        return self.predict(X)


@auto_block
class ETRBlock(ExtraTreesRegressor):
    def transform(self, X):
        return self.predict(X)


@auto_block
class CatBoostRegressorBlock(CatBoostRegressor):
    def transform(self, X):
        return self.predict(X)


@auto_block
class XGBCRegressorBlock(XGBRegressor):
    def transform(self, X):
        return self.predict(X)

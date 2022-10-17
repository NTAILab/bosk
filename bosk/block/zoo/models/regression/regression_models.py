from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)

from bosk.block import auto_block


@auto_block
class RFRBlock(RandomForestRegressor):
    def __init__(self, seed):
        super().__init__()
        self.random_state = seed

    def transform(self, X):
        return self.predict(X)


@auto_block
class ETRBlock(ExtraTreesRegressor):
    def __init__(self, seed):
        super().__init__()
        self.random_state = seed

    def transform(self, X):
        return self.predict(X)


@auto_block
class CatBoostRegressorBlock(CatBoostRegressor):
    def __init__(self, *args):
        super().__init__(*args)

    def transform(self, X):
        return self.predict(X)


@auto_block
class XGBCRegressorBlock(XGBRegressor):
    def __init__(self, *args):
        super().__init__(*args)

    def transform(self, X):
        return self.predict(X)

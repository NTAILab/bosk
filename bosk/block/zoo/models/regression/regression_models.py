from bosk.block import auto_block
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
)


@auto_block
class RFRBlock(RandomForestRegressor):
    def __init__(self, seed):
        super().__init__()
        self.random_state = seed

    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class ETRBlock(ExtraTreesRegressor):
    def __init__(self, seed):
        super().__init__()
        self.random_state = seed

    def transform(self, X):
        return self.predict_proba(X)

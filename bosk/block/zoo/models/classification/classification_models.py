from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from bosk.block import auto_block


@auto_block
class RFCBlock(RandomForestClassifier):
    def __init__(self, seed):
        super().__init__()
        self.random_state = seed

    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class ETCBlock(ExtraTreesClassifier):
    def __init__(self, seed):
        super().__init__()
        self.random_state = seed

    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class CatBoostClassifierBlock(CatBoostClassifier):
    def __init__(self, *args):
        super().__init__(*args)

    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class XGBClassifierBlock(XGBClassifier):
    def __init__(self, *args):
        super().__init__(*args)

    def transform(self, X):
        return self.predict_proba(X)

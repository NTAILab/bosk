from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)

from bosk.block import auto_block


@auto_block
class RFCBlock(RandomForestClassifier):
    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class ETCBlock(ExtraTreesClassifier):
    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class CatBoostClassifierBlock(CatBoostClassifier):
    def transform(self, X):
        return self.predict_proba(X)


@auto_block
class XGBClassifierBlock(XGBClassifier):
    def transform(self, X):
        return self.predict_proba(X)

from bosk.block import auto_block
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
)


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

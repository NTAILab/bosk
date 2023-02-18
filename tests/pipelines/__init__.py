from .casual_func_test import CasualFuncForestTest as CasualFuncForest
from .casual_manual_test import CasualManualForestTest as CasualManualForest

# renaming is needed to exclude those imported classes
# from the pytest test discovery mechanism and avoid
# repeat of these tests in this way

__all__ = [
    "CasualFuncForest",
    "CasualManualForest",
]

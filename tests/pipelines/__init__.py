from .casual_func_test import CasualFuncForestTest as CasualFuncForest
from .casual_manual_test import CasualManualForestTest as CasualManualForest
from .conf_screen_test import ConfScreenForestTest as ConfScreenForest
from .weighted_cs_test import WeightedCSForestTest as WeightedCSForest
from .mg_scanning_test import MGScanning1DTest as MGScanning1DForest, MGScanning2DTest as MGScanning2DForest

# renaming is needed to exclude those imported classes
# from the pytest test discovery mechanism and avoid
# repeat of these tests in this way

__all__ = [
    "CasualFuncForest",
    "CasualManualForest",
    "ConfScreenForest",
    "WeightedCSForest",
    "MGScanning1DForest",
    "MGScanning2DForest",
]

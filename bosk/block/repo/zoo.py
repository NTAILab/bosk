from types import ModuleType
from ..base import BaseBlock
from .scope import ScopeBlockClassRepository
from typing import Type, Union
# from ..zoo.data_conversion.data_conversion import *
# from ..zoo.data_weighting.data_weighting import *
# from ..zoo.input_plugs.input_plugs import *
# from ..zoo.metrics.metrics import *
# from ..zoo.models.classification.classification_models import *
# from ..zoo.models.regression.regression_models import *
# from ..zoo.multi_grained_scanning.multi_grained_scanning import *
# from ..zoo.multi_grained_scanning.multi_grained_scanning_1d import *
# from ..zoo.multi_grained_scanning.multi_grained_scanning_2d import *
# from ..zoo.routing.routing import *

from ...block import zoo as zoo_module
import importlib
import pkgutil


def import_submodules_contents(module: ModuleType):
    """ Import contents of all submodules.

    Source:

    Args:
        module: Module.

    Returns:
        Dictionary of imported submodules contents.

    """
    results = {**module.__dict__}
    for _loader, name, is_package in pkgutil.walk_packages(module.__path__):
        full_name = module.__name__ + '.' + name
        cur_module = importlib.import_module(full_name)
        results.update(cur_module.__dict__)
        if is_package:
            results.update(import_submodules_contents(cur_module))
    return results


class ZooBlockClassRepository(ScopeBlockClassRepository):
    """Block class repository that extracts block classes from
    `block.zoo` submodule.
    """
    def __init__(self):
        # super().__init__(globals())
        zoo_contents = import_submodules_contents(zoo_module)
        super().__init__(zoo_contents)

from types import ModuleType
from ..base import BaseBlock
from .scope import ScopeBlockClassRepository
from typing import Type, Union
from ...block import zoo as zoo_module
import importlib
import pkgutil


IGNORE_IMPORT_ERROR_PACKAGES = ['jax']


def import_submodules_contents(module: ModuleType):
    """Import contents of all submodules.

    Source:

    Args:
        module: Module.

    Returns:
        Dictionary of imported submodules contents.

    """
    results = {**module.__dict__}
    for _loader, name, is_package in pkgutil.walk_packages(module.__path__):
        full_name = module.__name__ + '.' + name
        try:
            cur_module = importlib.import_module(full_name)
        except ModuleNotFoundError as ex:
            if ex.name in IGNORE_IMPORT_ERROR_PACKAGES:
                continue
            else:
                raise ex
        results.update(cur_module.__dict__)
        if is_package:
            results.update(import_submodules_contents(cur_module))
    return results


class ZooBlockClassRepository(ScopeBlockClassRepository):
    """Block class repository that extracts block classes from
    `block.zoo` submodule.

    """
    def __init__(self):
        zoo_contents = import_submodules_contents(zoo_module)
        super().__init__(zoo_contents)

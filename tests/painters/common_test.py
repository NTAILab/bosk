"""Script that contains common tests for all painters."""

from bosk.painter import *
from . import PIC_SAVE_DIR, PIC_SAVE_FMT
from ..pipelines import CasualFuncForest
from ..utility import get_all_subclasses
from os.path import isfile
import logging


def get_pipeline_wrap():
    """Function that returns the pipeline wrapper to be painted
    by the painters in the `all_painters_test`."""
    return CasualFuncForest()


def all_painters_test():
    """Common test for the painters, that checks the work of the `from_pipeline` method.
    To make the painter available to be discovered by this test,
    you should add it to the `__all__` variable of the `bosk.painter` package.
    Test checks if the images were rendered. The semantic must be checked by the user manually.
    The output directory and format are defined in the `bosk.tests.painters` package
    as the global constants `PIC_SAVE_DIR` and `PIC_SAVE_FMT`.
    """
    painter_cls_list = get_all_subclasses(BasePainter)
    logging.info('Following classes were found for the test: %r',
                 [p.__name__ for p in painter_cls_list])
    pipeline = get_pipeline_wrap().get_pipeline()
    for painter_cls in painter_cls_list:
        painter = painter_cls()
        painter.from_pipeline(pipeline)
        filename = f'{PIC_SAVE_DIR}/{painter_cls.__name__}_graph'
        painter.render(filename, PIC_SAVE_FMT)
        assert isfile(filename + f'.{PIC_SAVE_FMT}'), "The pipeline wasn't rendered"
        logging.info('Rendered the graph, please see and check "%s"', filename)

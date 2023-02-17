from bosk.painter import *
from . import PIC_SAVE_DIR, PIC_SAVE_FMT
from ..pipelines import CasualFuncForest
from ..utility import get_all_subclasses
from os.path import isfile
import logging


def get_pipeline_wrap():
    return CasualFuncForest()


def all_painters_test():
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

from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage
from .pipeline_getter import BasePipelineGetter
from .pipelines import *
import logging

def test_pipelines():
    pl_getter_cls = BasePipelineGetter.__subclasses__()
    logging.info('Following classes for the test were found: ' + format([cls.__name__ for cls in pl_getter_cls]))
    for plg_cls in pl_getter_cls:
        plg = plg_cls()
        pipeline = plg.get_pipeline()
        data = plg.get_fit_data()
        executor = RecursiveExecutor(pipeline, HandlingDescriptor.from_classes(Stage.FIT))
        executor(data)
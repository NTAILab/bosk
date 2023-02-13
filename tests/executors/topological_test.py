from abc import ABC, abstractmethod
from bosk.executor.topological import TopologicalExecutor
from bosk.pipeline.base import BasePipeline, Connection
from bosk.painter.topological import TopologicalPainter
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage
from ..pipelines.base import PipelineTestBase
from ..utility import fit_pipeline
from collections import defaultdict
import numpy as np
import logging
from ..pipelines import CasualManualForestTest
from os.path import isfile
from typing import Set, List, Tuple, Dict
from bosk.block.base import BaseBlock
from bosk.block.zoo.data_conversion import AverageBlock


class BaseTopSortChecker(ABC):
    def __init__(self) -> None:
        super().__init__()

    def get_stage(self) -> Stage:
        return Stage.FIT

    @abstractmethod
    def get_pipeline(self) -> BasePipeline:
        ...

    @abstractmethod
    def get_sufficient_blocks(self) -> Set[BaseBlock]:
        ...

    @abstractmethod
    def get_order_requirements(self) -> List[Tuple[BaseBlock]]:
        """Get list of ordered chains, that must appear in the topological order.
        For example, chain (Block1, Block2, Block3) means that the topological 
        order must contain Block1 < Block2 < Block3 subsequence."""

    def check_blocks_presence(self, top_order: List[BaseBlock]) -> bool:
        return self.get_sufficient_blocks() == set(top_order)

    def top_list_to_dict(self, top_order: List[BaseBlock]) -> Dict[BaseBlock, int]:
        idx_dict = dict()
        for i, block in enumerate(top_order):
            assert block not in idx_dict, \
                f'Block {format(block)} appears more than once in the topological order'
            idx_dict[block] = i
        return idx_dict

    def check_order(self, top_order: List[BaseBlock]) -> bool:
        chains = self.get_order_requirements()
        top_dict = self.top_list_to_dict(top_order)
        for chain in chains:
            for i in range(len(chain) - 1):
                if top_dict[chain[i]] >= top_dict[chain[i + 1]]:
                    return False
        return True

    def check_topological_sort(self, top_order: List[BaseBlock]) -> None:
        assert self.check_blocks_presence(top_order), \
            'Topological sort produced the wrong set of blocks'
        assert self.check_order(top_order), \
            'Topological sort produced the wrong blocks order'


class StraightTopSortChecker(BaseTopSortChecker):
    def __init__(self) -> None:
        super().__init__()
        nodes = [AverageBlock() for _ in range(5)]
        conns = [Connection(nodes[i - 1].slots.outputs['output'],
                            nodes[i].slots.inputs['X']) for i in range(1, 5)]
        self.pipeline = BasePipeline(nodes, conns,
                                     {'X': nodes[0].slots.inputs['X']},
                                     {'X': nodes[-1].slots.outputs['output']})

    def get_sufficient_blocks(self) -> Set[BaseBlock]:
        return set(self.pipeline.nodes)

    def get_pipeline(self) -> BasePipeline:
        return self.pipeline

    def get_order_requirements(self) -> List[Tuple[BaseBlock]]:
        return [tuple(self.pipeline.nodes)]


class TopologicalExecTest():

    def get_pw_to_paint(self):
        return CasualManualForestTest()

    def painter_test(self):
        filename = 'topological_painter'
        dirname = 'tests/pictures'
        painter = TopologicalPainter()
        pip_wrapper = self.get_pw_to_paint()
        pipeline = pip_wrapper.get_pipeline()
        fit_executor = TopologicalExecutor(pipeline,
                                           HandlingDescriptor.from_classes(Stage.FIT), *pip_wrapper.get_fit_in_out())
        painter.from_executor(fit_executor)
        fit_filename = f'{dirname}/{filename}_fit.png'
        painter.render(fit_filename)
        assert isfile(fit_filename), "Fit pipeline wasn't rendered"
        logging.info('Rendered the fit graph, please see and check "%s"', fit_filename)
        painter = TopologicalPainter()
        tf_executor = TopologicalExecutor(pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                                          *pip_wrapper.get_transform_in_out())
        painter.from_executor(tf_executor)
        tf_filename = f'{dirname}/{filename}_transform.png'
        painter.render(tf_filename)
        assert isfile(tf_filename), "Transform pipeline wasn't rendered"
        logging.info('Rendered the transform graph, please see and check "%s"', tf_filename)

    def topological_sort_test(self):
        tests_cls = BaseTopSortChecker.__subclasses__()
        logging.info('Following classes were found for the test: %r',
                     [t.__name__ for t in tests_cls])
        for test_cls in tests_cls:
            test = test_cls()
            pipeline = test.get_pipeline()
            inp_blocks = [slot.parent_block for slot in pipeline.inputs.values()]
            executor = TopologicalExecutor(pipeline,
                                           HandlingDescriptor.from_classes(test.get_stage()))
            top_sort_order = executor._topological_sort(executor._get_forward_aj_list(), inp_blocks)
            test.check_topological_sort(top_sort_order)

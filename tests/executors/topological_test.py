from abc import ABC, abstractmethod
from bosk.executor.topological import TopologicalExecutor
from bosk.pipeline.base import BasePipeline, Connection
from bosk.painter.topological import TopologicalPainter
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage
from ..utility import get_all_subclasses, connect_chain
from ..painters import PIC_SAVE_DIR, PIC_SAVE_FMT
import logging
from ..pipelines import CasualManualForest
from os.path import isfile
from typing import Set, List, Dict
from bosk.block.base import BaseBlock
from bosk.block.zoo.data_conversion import AverageBlock, ArgmaxBlock, ConcatBlock


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
    def get_order_requirements(self) -> List[List[BaseBlock]]:
        """Get list of ordered chains, that must appear in the topological order.
        For example, chain [Block1, Block2, Block3] means that the topological 
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
        conns = connect_chain(nodes)
        self.pipeline = BasePipeline(nodes, conns,
                                     {'X': nodes[0].slots.inputs['X']},
                                     {'X': nodes[-1].slots.outputs['output']})

    def get_sufficient_blocks(self) -> Set[BaseBlock]:
        return set(self.pipeline.nodes)

    def get_pipeline(self) -> BasePipeline:
        return self.pipeline

    def get_order_requirements(self) -> List[List[BaseBlock]]:
        return [self.pipeline.nodes]


class SplittedTopSortChecker(BaseTopSortChecker):
    def __init__(self) -> None:
        super().__init__()
        branch_1 = [AverageBlock() for _ in range(3)]
        branch_2 = [ArgmaxBlock() for _ in range(5)]
        branch_3 = [AverageBlock(), ArgmaxBlock()]
        head = [ArgmaxBlock(), AverageBlock(), ArgmaxBlock()]
        nodes = branch_1 + branch_2 + branch_3 + head
        br_1_conns = connect_chain(branch_1)
        br_2_conns = connect_chain(branch_2)
        br_3_conns = connect_chain(branch_3)
        head_conns = connect_chain(head)
        split_conns = [Connection(head[-1].slots.outputs['output'], branch_1[0].slots.inputs['X']),
                       Connection(head[-1].slots.outputs['output'], branch_2[0].slots.inputs['X']),
                       Connection(head[-1].slots.outputs['output'], branch_3[0].slots.inputs['X'])]
        conns = br_1_conns + br_2_conns + br_3_conns + head_conns + split_conns
        self.pipeline = BasePipeline(nodes, conns,
                                     {'X': head[0].slots.inputs['X']},
                                     {'branch_1': branch_1[-1].slots.outputs['output'],
                                         'branch_2': branch_2[-1].slots.outputs['output'],
                                         'branch_3': branch_3[-1].slots.outputs['output']})
        self.head = head
        self.branch_1, self.branch_2, self.branch_3 = branch_1, branch_2, branch_3

    def get_sufficient_blocks(self) -> Set[BaseBlock]:
        return set(self.pipeline.nodes)

    def get_pipeline(self) -> BasePipeline:
        return self.pipeline

    def get_order_requirements(self) -> List[List[BaseBlock]]:
        chain_1 = self.head + self.branch_1
        chain_2 = self.head + self.branch_2
        chain_3 = self.head + self.branch_3
        return [chain_1, chain_2, chain_3]


class MergedTopSortChecker(BaseTopSortChecker):
    def __init__(self) -> None:
        super().__init__()
        branch_1 = [AverageBlock() for _ in range(5)]
        branch_2 = [ArgmaxBlock() for _ in range(3)]
        merged_chain = [ConcatBlock(['X_1', 'X_2'])] + [AverageBlock(), ArgmaxBlock()]
        nodes = branch_1 + merged_chain + branch_2
        br_1_con = connect_chain(branch_1)
        br_2_con = connect_chain(branch_2)
        merged_chain_con = connect_chain(merged_chain)
        merge_con = [Connection(branch_1[-1].slots.outputs['output'], merged_chain[0].slots.inputs['X_1']),
                     Connection(branch_2[-1].slots.outputs['output'], merged_chain[0].slots.inputs['X_2'])]
        conns = merge_con + br_1_con + merged_chain_con + br_2_con
        self.pipeline = BasePipeline(nodes, conns,
                                     {'branch_1': branch_1[0].slots.inputs['X'],
                                         'branch_2': branch_2[0].slots.inputs['X']},
                                     {'output': merged_chain[-1].slots.outputs['output']})
        self.branch_1, self.branch_2 = branch_1, branch_2
        self.merged_chain = merged_chain

    def get_sufficient_blocks(self) -> Set[BaseBlock]:
        return set(self.pipeline.nodes)

    def get_pipeline(self) -> BasePipeline:
        return self.pipeline

    def get_order_requirements(self) -> List[List[BaseBlock]]:
        chain_1 = self.branch_1 + self.merged_chain
        chain_2 = self.branch_2 + self.merged_chain
        return [chain_1, chain_2]


class FakeNodesTopSortChecker(SplittedTopSortChecker):
    def __init__(self) -> None:
        super().__init__()
        self.pipeline = BasePipeline(
            self.pipeline.nodes, self.pipeline.connections,
            {'X': self.head[-1].slots.inputs['X']},
            {'branch_1': self.branch_1[-1].slots.outputs['output'],
             'branch_2': self.branch_2[-1].slots.outputs['output'],
             'branch_3': self.branch_3[-1].slots.outputs['output']}
        )

    def get_sufficient_blocks(self) -> Set[BaseBlock]:
        return set([self.head[-1]] + self.branch_1 + self.branch_2 + self.branch_3)

    def get_order_requirements(self) -> List[List[BaseBlock]]:
        return [[self.head[-1]] + self.branch_1,
                [self.head[-1]] + self.branch_2,
                [self.head[-1]] + self.branch_3]


class DFSChecker():
    def __init__(self) -> None:
        branch_1 = [AverageBlock() for _ in range(3)]
        branch_2 = [ArgmaxBlock() for _ in range(5)]
        branch_3 = [AverageBlock(), ArgmaxBlock()]
        head = [ArgmaxBlock(), AverageBlock(), ArgmaxBlock()]
        nodes = branch_1 + branch_2 + branch_3 + head
        br_1_conns = connect_chain(branch_1)
        br_2_conns = connect_chain(branch_2)
        br_3_conns = connect_chain(branch_3)
        head_conns = connect_chain(head)
        split_conns = [Connection(head[-1].slots.outputs['output'], branch_1[0].slots.inputs['X']),
                       Connection(head[-1].slots.outputs['output'], branch_2[0].slots.inputs['X']),
                       Connection(head[-1].slots.outputs['output'], branch_3[0].slots.inputs['X'])]
        conns = br_1_conns + br_2_conns + br_3_conns + head_conns + split_conns
        self.pipeline = BasePipeline(nodes, conns,
                                     {'X': head[-1].slots.inputs['X']},
                                     {'branch_2': branch_2[-3].slots.outputs['output']})
        self.head = head
        self.branch_2 = branch_2

    def get_pipeline(self):
        return self.pipeline

    def get_sufficient_blocks(self):
        return set([self.head[-1]] + self.branch_2[:-2])


class TopologicalExecTest():

    def get_pw_to_paint(self):
        return CasualManualForest()

    def painter_test(self):
        filename = 'topological_painter'
        dirname = PIC_SAVE_DIR
        painter = TopologicalPainter()
        pip_wrapper = self.get_pw_to_paint()
        pipeline = pip_wrapper.get_pipeline()
        fit_executor = TopologicalExecutor(pipeline,
                                           HandlingDescriptor.from_classes(Stage.FIT),
                                           *pip_wrapper.get_fit_in_out())
        painter.from_executor(fit_executor)
        fit_filename = f'{dirname}/{filename}_fit'
        painter.render(fit_filename, PIC_SAVE_FMT)
        assert isfile(fit_filename + f'.{PIC_SAVE_FMT}'), "Fit pipeline wasn't rendered"
        logging.info('Rendered the fit graph, please see and check "%s"', fit_filename)
        painter = TopologicalPainter()
        tf_executor = TopologicalExecutor(pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                                          *pip_wrapper.get_transform_in_out())
        painter.from_executor(tf_executor)
        tf_filename = f'{dirname}/{filename}_transform'
        painter.render(tf_filename, PIC_SAVE_FMT)
        assert isfile(tf_filename + f'.{PIC_SAVE_FMT}'), "Transform pipeline wasn't rendered"
        logging.info('Rendered the transform graph, please see and check "%s"', tf_filename)

    def topological_sort_test(self):
        tests_cls = get_all_subclasses(BaseTopSortChecker)
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

    def dfs_test(self):
        checker = DFSChecker()
        pipeline = checker.get_pipeline()
        executor = TopologicalExecutor(pipeline,
                                       HandlingDescriptor.from_classes(Stage.FIT))
        input_blocks = [slot.parent_block for slot in pipeline.inputs.values()]
        output_blocks = [slot.parent_block for slot in pipeline.outputs.values()]
        forward_pass = executor._dfs(executor._get_forward_aj_list(), input_blocks)
        backward_pass = executor._dfs(executor._get_backward_aj_list(), output_blocks)
        assert forward_pass & backward_pass == checker.get_sufficient_blocks(), \
            "Pipeline optimization with the dfs produced a wrong blocks' set"

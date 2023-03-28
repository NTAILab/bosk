from abc import ABC, abstractmethod
from bosk.block.base import BaseBlock
from bosk.block.slot import BlockInputSlot, BlockOutputSlot
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.pipeline.base import BasePipeline, Connection
from bosk.data import BaseData
from collections import deque, defaultdict
from copy import deepcopy
from .metric import BaseMetric
import joblib
from functools import cache
from typing import List, Set, Dict, Optional, Iterable


class BaseForeignModel(ABC):
    """Adapter class for all models, defined outside
    of the `bosk` framework. It is needed to make sure
    that the model handles bosk's style of the data transmission. 
    """

    @abstractmethod
    def fit(self, data: Dict[str, BaseData]) -> None:
        """Method to handle the data dictionary and fit the model."""

    @abstractmethod
    def predict(self, data: Dict[str, BaseData]) -> Dict[str, BaseData]:
        """Method for using the fitted model and obtain transformed 
        data dictionary."""

    @abstractmethod
    def set_random_state(self, random_state: int) -> None:
        """Set random state for the model."""


@cache
def get_block_hash(block: BaseBlock):
    """Helper function to obtain blocks' hashes
    and cache them.
    """
    return joblib.hash(block)


class BaseComparator(ABC):
    """Class that performes comparison of different models.
    The models, defined via `bosk` framework, marked as `pipelines`
    and may have a common part in them to optimize calculations
    (common part pipeline will be executed once and retreived 
    data will be used in other pipelines). The common part must be
    a common begining of all pipelines.

    The models, defined with others but `bosk` frameworks, are
    marked as `models` and must be wrapped in `BaseForeignModel`
    adapter to handle the `bosk` data transmission style.
    """

    random_state: int

    def _get_results_dict(self, nested_dict:
                          Dict[str, float | List[float]]) -> Dict[str, Dict[str, float | List[float]]]:
        res_dict = dict()
        for i in range(len(self.optim_pipelines)):
            res_dict[f'pipeline_{i}'] = deepcopy(nested_dict)
        for i in range(len(self.models)):
            res_dict[f'model_{i}'] = deepcopy(nested_dict)
        return res_dict

    def _get_aj_lists(self, pipeline: BasePipeline):
        conn_map_blocks: Dict[BaseBlock, Set[BaseBlock]] = defaultdict(set)  # output->input, forward pass
        conn_map_conns: Dict[BlockInputSlot, BlockOutputSlot] = dict()  # input->output, backward pass
        for conn in pipeline.connections:
            conn_map_blocks[conn.src.parent_block].add(conn.dst.parent_block)
            conn_map_conns[conn.dst] = conn.src
        return conn_map_blocks, conn_map_conns

    def _compare_blocks(self, block_cp: BaseBlock, block_pip: BaseBlock, aj_list_cp, aj_list_pip, blocks_iso):
        for inp_name, inp_slot in block_cp.slots.inputs.items():
            pip_slot = block_pip.slots.inputs[inp_name]
            cp_corr_out_slot = aj_list_cp.get(inp_slot, None)
            if cp_corr_out_slot is None:
                # in common part the slot is not used, but it's used in the pipeline
                if pip_slot in aj_list_pip:
                    return False
            else:
                pip_corr_out_slot = aj_list_pip.get(pip_slot, None)
                # in common part the slot is used, but it's not used in the pipeline
                if pip_corr_out_slot is None:
                    return False
                cp_prev_block = cp_corr_out_slot.parent_block
                pip_prev_block = pip_corr_out_slot.parent_block
                # check if the corresponding block of the common part
                # is isomorphic to the corresponding block of the pipeline
                return blocks_iso[cp_prev_block] == pip_prev_block
        return True

    def __init__(self, pipelines: Optional[BasePipeline | List[BasePipeline]],
                 common_part: Optional[BasePipeline],
                 foreign_models: Optional[BaseForeignModel | List[BaseForeignModel]],
                 random_state: Optional[int] = None) -> None:

        # boundary cases
        if pipelines is None and foreign_models is None:
            raise RuntimeError("You must select models to compare")
        if pipelines is not None and not isinstance(pipelines, Iterable):
            pipelines = [pipelines]
        if foreign_models is not None and not isinstance(foreign_models, Iterable):
            foreign_models = [foreign_models]
        if (pipelines is None or len(pipelines) == 1) and common_part is not None:
            raise RuntimeError("You must select multiple pipelines to use common part optimization")

        self.random_state = random_state
        self.models = [] if foreign_models is None else foreign_models
        for model in self.models:
            model.set_random_state(random_state)
        if pipelines is not None:
            for pipeline in pipelines:
                    pipeline.set_random_state(random_state)
        if common_part is None:
            self.common_pipeline = None
            self.optim_pipelines = [] if pipelines is None else pipelines
            return
        self.common_pipeline = common_part
        self.common_pipeline.set_random_state(random_state)

        # pipelines' optimization process
        # 1. Step of bfs in common part is leading
        # 2. Each time when block is extracted on step 1, we look in the queue
        # of other pipelines for isomorphic blocks (compare blocks + conns)
        # 3. If the isomorphic block is found for i'th pipeline, then we
        # add his neighbour blocks in his queue
        # 4. If we didn't find the isomophic block, then we should write
        # a warning and exclude the pipeline from further comparison,
        # because the isomorphic block can't be on another level.
        # 5. Purpose of the algorithm - build two isomorphisms: for the blocks and
        # for the connections. When the last block in common part's bfs is proceeded,
        # we will be able to map common part's outputs and slots in other pipelines.
        # 6. If the common part fully matches with the pipeline, we add an input plugs
        # to the pipeline and redirect all outputs to this plugs. Type of plug is defined
        # by the original output block.
        # 7. Otherwise, we disconnect the original block (isomorphic for the common's part
        # output) from the corresponding block in the pipeline and mark correspondig input
        # slot as new input of the pipeline (we will pass data from the common's part to
        # this input).

        conn_map_blocks_cp, conn_map_conns_cp = self._get_aj_lists(common_part)
        begin_blocks_cp = [inp_conn.parent_block for inp_conn in common_part.inputs.values()]

        conn_maps_blocks = []
        conn_maps_conns = []
        visited = []
        queue_list = []
        block_iso_list = []
        conn_iso_list = []

        for pipeline in pipelines:
            cur_blocks_al, cur_conns_al = self._get_aj_lists(pipeline)
            conn_maps_blocks.append(cur_blocks_al)
            conn_maps_conns.append(cur_conns_al)
            cur_begin_blocks = [inp_conn.parent_block for inp_conn in pipeline.inputs.values()]
            visited.append(set(cur_begin_blocks))
            queue_list.append(deque(cur_begin_blocks))
            block_iso_list.append(dict())  # map from common part's blocks to model's
            conn_iso_list.append(dict())  # map from common part's conns to model's

        # bfs
        visited_cp = set(begin_blocks_cp)
        queue_cp = deque(begin_blocks_cp)
        while queue_cp:
            cur_block = queue_cp.popleft()
            cur_block_hash = get_block_hash(cur_block)
            for i in range(len(pipelines)):
                queue_hashes = [get_block_hash(block) for block in queue_list[i]]
                # find matching block in queue
                match_idxes = [i for i, hash in enumerate(queue_hashes) if hash == cur_block_hash]
                match_block: BaseBlock = None
                for idx in match_idxes:
                    if self._compare_blocks(cur_block, queue_list[i][idx],
                                            conn_map_conns_cp, conn_maps_conns[i], block_iso_list[i]):
                        match_block = queue_list[i][idx]
                        # delete matched block from the queue
                        del queue_list[i][idx]
                        break
                # if the matching block is not found, it's an error
                if match_block is None:
                    raise RuntimeError(f'Unable to find block {repr(cur_block)} in pipeline {i}')

                # building isomorphism
                block_iso_list[i][cur_block] = match_block
                for inp_name, inp_slot in cur_block.slots.inputs.items():
                    conn_iso_list[i][inp_slot] = match_block.slots.inputs[inp_name]
                for out_name, out_slot in cur_block.slots.outputs.items():
                    conn_iso_list[i][out_slot] = match_block.slots.outputs[out_name]

                # add neigbour nodes to the queue
                for neig_node in conn_maps_blocks[i][match_block]:
                    if neig_node not in visited[i]:
                        visited[i].add(neig_node)
                        queue_list[i].append(neig_node)

            # continue bfs for the common part
            for neig_node in conn_map_blocks_cp[cur_block]:
                if neig_node not in visited_cp:
                    visited_cp.add(neig_node)
                    queue_cp.append(neig_node)

        # redefining input pipelines to use calculations
        # from the common part for them
        self.optim_pipelines: List[BasePipeline] = []
        for i, pipeline in enumerate(pipelines):
            # All common part's inputs, must have isomorphic
            # slots in the pipeline and should NOT be excluded.
            new_inputs = pipeline.inputs.copy()
            for inp_name, inp_slot in common_part.inputs.items():
                iso_slot = conn_iso_list[i].get(inp_slot, None)
                assert iso_slot is not None, \
                    "Each input in the common part must have corresponding input in the pipeline"
                assert iso_slot == pipeline.inputs[inp_name], \
                    "Inputs in common part and pipeline must have the same name"
                # we shouldn't delete input, because it can be used in other parts but the common one
                # del new_inputs[inp_name]
            # The common part's output slot must have isomorphic
            # slot in the pipeline, for which we should make additional input.
            # Need to account that we may have common output
            # We also need to delete the original connection
            output_slots: Dict[BlockOutputSlot, str] = dict()
            for name, slot in pipeline.outputs.items():
                output_slots[slot] = name
            new_outputs = pipeline.outputs.copy()
            # unable to make output slot directly as input,
            # so we need to add input blocks
            extra_blocks: List[BaseBlock] = []
            extra_conns: List[Connection] = []
            conns_to_remove: Set[BlockInputSlot] = set()
            for out_name, out_slot in common_part.outputs.items():
                iso_slot = conn_iso_list[i].get(out_slot, None)
                assert iso_slot is not None, \
                    "Each output in the common part must have corresponding slot in the pipeline"
                assert out_name not in new_inputs, \
                    "Names of the common part's outputs must not intersect with the pipeline's inputs"
                # we should add input block for each common part's output
                # to make easier connection of multiple blocks to one output
                if iso_slot.meta.stages.transform:
                    input_plug = InputBlock()
                else:
                    input_plug = TargetInputBlock()
                extra_blocks.append(input_plug)
                new_inputs[out_name] = next(iter(input_plug.slots.inputs.values()))
                plug_out_slot = next(iter(input_plug.slots.outputs.values()))
                # case when pipeline's output corresponds to common part's output
                if iso_slot in output_slots:
                    new_outputs[output_slots[iso_slot]] = plug_out_slot
                    continue
                # case when corresponding slot is in the middle of the pipeline
                # finding corresponding input slots
                for in_slot, corr_out_slot in conn_maps_conns[i].items():
                    if corr_out_slot == iso_slot:
                        extra_conns.append(Connection(plug_out_slot, in_slot))
                        conns_to_remove.add(in_slot)
            new_conns = extra_conns
            for conn in pipeline.connections:
                if conn.dst not in conns_to_remove:
                    new_conns.append(conn)
            new_blocks = pipeline.nodes + extra_blocks
            self.optim_pipelines.append(BasePipeline(new_blocks, new_conns, new_inputs, new_outputs))

    @abstractmethod
    def get_score(self, data: Dict[str, BaseData],
                  metrics: List[BaseMetric], random_) -> Dict[str, Dict[str, float | List[float]]]:
        """Function to obtain results of different metrics for the models.
        Returns:
            Dictionary with keys as models' names (`pipeline_i` for i-th pipeline and 
            `model_i` for i-th foreign model) and values as dictionaries. These dictionaries
            contain metrics' names as keys (metric name for the named metrics and `metric_i`
            for the unnamed ones) and scores as values. List length of n corresponds to n
            iterations or folds.
        """

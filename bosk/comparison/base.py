from abc import ABC, abstractmethod
from bosk.block.base import BaseBlock
from bosk.block.slot import BlockInputSlot, BlockOutputSlot
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.pipeline.base import BasePipeline, Connection
from bosk.block.base import BaseInputBlock
from bosk.block.slot import BaseSlot
from bosk.data import BaseData
from collections import deque, defaultdict
from .metric import BaseMetric
import joblib
from functools import cache
from typing import List, Set, Dict, Optional, Iterable, Deque, Tuple
from copy import deepcopy
from pandas import DataFrame


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
def get_block_md5_hash(block: BaseBlock):
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

    def _get_aj_lists(self, pipeline: BasePipeline):
        # output->input, forward pass
        conn_map_blocks: Dict[BaseBlock, Set[BaseBlock]] = defaultdict(set)
        # input->output, backward pass
        conn_map_conns: Dict[BlockInputSlot, BlockOutputSlot] = dict()
        for conn in pipeline.connections:
            conn_map_blocks[conn.src.parent_block].add(conn.dst.parent_block)
            conn_map_conns[conn.dst] = conn.src
        return conn_map_blocks, conn_map_conns

    def _compare_blocks_conns(self, block_1: BaseBlock, block_2: BaseBlock,
                        aj_list_1, aj_list_2, conns_iso_1, conns_iso_2) -> bool:
        # conns_iso: slot isomorphism from pipeline to
        # common part. It's needed to ensure that
        # connections to blocks are headed out of the
        # same block (in terms of isomorphism)
        for inp_name, inp_slot_1 in block_1.slots.inputs.items():
            inp_slot_2 = block_2.slots.inputs[inp_name]
            corr_out_1 = aj_list_1.get(inp_slot_1, None)
            corr_out_2 = aj_list_2.get(inp_slot_2, None)
            is_slot_used = (corr_out_1 is not None, corr_out_2 is not None)
            # in one pipeline the slot is used, but in the other is not
            if is_slot_used[0] ^ is_slot_used[1]:
                return False
            else:
                # both slots are not used
                if not is_slot_used[0] and not is_slot_used[1]:
                    continue
                # else check prev blocks
                # if we don't have prev block
                # in the common part, then the bfs didn't proceeded prev block yet
                iso_out_conn_1 = conns_iso_1.get(corr_out_1, None)
                iso_out_conn_2 = conns_iso_2.get(corr_out_2, None)
                if iso_out_conn_1 is None or iso_out_conn_2 is None:
                    return False
                common_prev_block_1 = iso_out_conn_1.parent_block
                common_prev_block_2 = iso_out_conn_2.parent_block
                if common_prev_block_1 != common_prev_block_2:
                    return False
        return True

    def _set_random_state(self, random_state=None):
        for model in self.models:
            model.set_random_state(random_state)
        for pipeline in self._optim_pipelines:
            pipeline.set_random_state(random_state)
        if self._common_pipeline is not None:
            self._common_pipeline.set_random_state(random_state)

    def _set_manual_state(self, pipeline, random_state):
        for node in pipeline.nodes:
            node.set_random_state(random_state)

    def _get_common_inputs(self, pipelines: List[BasePipeline]) -> List[str]:
        input_names = []
        input_dicts = [pip.inputs for pip in pipelines]
        test_dict = input_dicts[0]
        other_dicts = input_dicts[1:]
        for inp_name in test_dict:
            f_common = True
            for d in other_dicts:
                if inp_name not in d:
                    f_common = False
                    break
            if f_common:
                input_names.append(inp_name)
        return input_names

    def _find_next_block(self, cur_block, leading_conn_map,
                         pipelines, conn_maps_list, queue_list,
                         leading_slots_iso, slots_iso_list) -> Optional[Tuple[List[BaseBlock], List[int]]]:
        matched_blocks: List[BaseBlock] = []
        matched_idx: List[int] = []
        cur_block_hash = get_block_md5_hash(cur_block)
        # cur_block is from the leading pipeline
        # and pipelines exclude the leading one
        for i in range(len(pipelines)):
            queue_hashes = [get_block_md5_hash(block) for block in queue_list[i]]
            # find matching block in queue
            match_idxes = [i for i, hash in enumerate(queue_hashes) if hash == cur_block_hash]
            match_block: BaseBlock = None
            for idx in match_idxes:
                if self._compare_blocks_conns(cur_block, queue_list[i][idx], leading_conn_map, 
                                              conn_maps_list[i], leading_slots_iso, slots_iso_list[i]):
                    match_block = queue_list[i][idx]

                    matched_blocks.append(match_block)
                    matched_idx.append(idx)
                    # delete matched block from the queue
                    # del queue_list[i][idx]
                    break
            # if the matching block is not found, it's an error
            if match_block is None:
                return None
        return matched_blocks, matched_idx

    def _add_common_block(self, pipelines_blocks: List[BaseBlock],
                   conn_map_list, iso_blocks_list,
                   iso_slots_list, common_blocks,
                   common_conn_map, added_list) -> None:
        block_for_copy = pipelines_blocks[0]
        new_block = deepcopy(block_for_copy)
        common_blocks.append(new_block)
        # add proper connections for a new block
        for inp_name, inp_slot in block_for_copy.slots.inputs.items():
            corr_output = conn_map_list[0].get(inp_slot, None)
            if corr_output is not None:
                iso_output = iso_slots_list[0][corr_output]
                iso_input = new_block.slots.inputs[inp_name]
                common_conn_map[iso_input] = iso_output
        # build isomorphisms
        # and add blocks to the 'visited' bfs set
        for i in range(len(pipelines_blocks)):
            iso_blocks_list[i][new_block] = pipelines_blocks[i]
            for inp_name, inp_slot in new_block.slots.inputs.items():
                iso_slots_list[i][pipelines_blocks[i].slots.inputs[inp_name]] = inp_slot
            for out_name, out_slot in new_block.slots.outputs.items():
                iso_slots_list[i][pipelines_blocks[i].slots.outputs[out_name]] = out_slot
            added_list[i].add(pipelines_blocks[i])

    def _continue_bfs(self, continue_blocks, queue_list, conn_maps_blocks, added_blocks_list) -> None:
        for i in range(len(continue_blocks)):
            idx_to_del = [idx for idx, block in enumerate(queue_list[i]) if block == continue_blocks[i]]
            for idx in idx_to_del:
                del queue_list[i][idx]
            neig_nodes = conn_maps_blocks[i].get(continue_blocks[i], None)
            if neig_nodes is None:
                continue
            for neig_node in neig_nodes:
                if neig_node not in added_blocks_list[i] and (not queue_list[i] or neig_node != queue_list[i][-1]):
                    queue_list[i].append(neig_node)

    def _get_input_plug(self, slot: BaseSlot) -> BaseInputBlock:
        if slot.meta.stages.transform:
            return InputBlock()
        return TargetInputBlock()

    def _splice_pipelines(self, conns_to_append,
                          conns_to_remove, inp_slot_pip,
                          common_outputs, out_slot_cp,
                          extra_inputs, extra_blocks) -> None:
        conns_to_remove.append(inp_slot_pip)
        corr_out_block_cp = out_slot_cp.parent_block
        # assert there're only one output slot in the block
        output_name = str(hash(corr_out_block_cp))
        # if we don't have common out for this block
        if output_name not in common_outputs:
            # add new common output
            common_outputs[output_name] = out_slot_cp
        # if we don't have input plug for this block
        if output_name not in extra_inputs:
            input_plug = self._get_input_plug(inp_slot_pip)
            extra_blocks.append(input_plug)
            extra_inputs[output_name] = input_plug.get_single_input()
        input_plug = extra_inputs[output_name].parent_block
        plug_out_slot = next(iter(input_plug.slots.outputs.values()))
        conns_to_append.append(Connection(plug_out_slot, inp_slot_pip))

    def _get_common_input_dict(self, common_inp_names, inp_dict_pip, slots_iso) -> Dict[str, BlockInputSlot]:
        inp_dict = dict()
        for name in common_inp_names:
            inp_dict[name] = slots_iso[inp_dict_pip[name]]
        return inp_dict

    def __init__(self, pipelines: Optional[BasePipeline | List[BasePipeline]],
                 foreign_models: Optional[BaseForeignModel | List[BaseForeignModel]],
                 f_optimize_pipelines: bool = True, random_state: Optional[int] = None) -> None:

        # boundary cases
        if pipelines is None and foreign_models is None:
            raise RuntimeError("You must select models to compare")
        if pipelines is not None and not isinstance(pipelines, Iterable):
            pipelines = [pipelines]
        if foreign_models is not None and not isinstance(foreign_models, Iterable):
            foreign_models = [foreign_models]

        self.random_state = random_state
        self.models = [] if foreign_models is None else foreign_models

        if not f_optimize_pipelines or pipelines is None:
            self._common_pipeline = None
            self._optim_pipelines = [] if pipelines is None else pipelines
            self._set_random_state(self.random_state)
            return

        pre_ran_state = 0  # fixed random state for proper joblib hash generation
        for pipeline in pipelines:
            self._set_manual_state(pipeline, pre_ran_state)

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

        # conn_map_blocks_cp, conn_map_conns_cp = self._get_aj_lists(common_part)
        # begin_blocks_cp = [inp_conn.parent_block for inp_conn in common_part.inputs.values()]

        conn_maps_blocks = []
        conn_maps_conns = []
        queue_list = []
        block_iso_list = []
        slot_iso_list = []
        added_blocks_list = []

        common_inputs_names = self._get_common_inputs(pipelines)
        common_blocks = []
        common_conn_map: Dict[BlockInputSlot, BlockOutputSlot] = dict()

        for pipeline in pipelines:
            cur_blocks_al, cur_conns_al = self._get_aj_lists(pipeline)
            conn_maps_blocks.append(cur_blocks_al)
            conn_maps_conns.append(cur_conns_al)
            cur_begin_blocks = [pipeline.inputs[name].parent_block for name in common_inputs_names]
            added_blocks_list.append(set())
            queue_list.append(deque(cur_begin_blocks))
            block_iso_list.append(dict())  # map from common part's blocks to model's
            slot_iso_list.append(dict())  # map from the original pipeline to the common one

        lead_queue: Deque[BaseBlock] = queue_list[0]
        other_queues = queue_list[1:]
        # lead_pipeline = pipelines[0]
        other_pipelines = pipelines[1:]
        lead_conn_map_conns = conn_maps_conns[0]
        other_conn_map_conns = conn_maps_conns[1:]
        lead_slot_iso = slot_iso_list[0]
        other_slot_iso_list = slot_iso_list[1:]

        # bfs
        # there is a opportunity for an
        # optimization: do the pipelines modification
        # within the bfs, when we didn't find any
        # matching blocks
        # IMPORTANT: the order of the block traversal in terms of
        # graph's depth is not defined: graph A -> B, A -> C, B -> C
        # leads to the traversals (A, B, C), (A, C, B)
        # the second traversal must be fixed.
        # We can do it either doing dfs for the leading pipeline
        # or replace the `visited` set in bfs with the added to
        # the common pipeline blocks. Also, once appended
        # in the common pipeline block must be excluded
        # from all bfs queues.
        while lead_queue:
            cur_block = lead_queue.popleft()
            matched_blocks = self._find_next_block(
                cur_block, lead_conn_map_conns, other_pipelines,
                other_conn_map_conns, other_queues, 
                lead_slot_iso, other_slot_iso_list
            )

            if matched_blocks is None:
                # no isomorphic blocks for current block
                # just skip it
                continue
            
            matched_idx = matched_blocks[1]
            matched_blocks = matched_blocks[0]

            # found matching blocks
            # need to extract them from queues,
            for i, queue_idx in enumerate(matched_idx):
                del other_queues[i][queue_idx]

            # add a block in the common pipeline
            # and build the isomorhism
            pipelines_blocks = [cur_block] + matched_blocks
            self._add_common_block(pipelines_blocks, conn_maps_conns, block_iso_list,
                            slot_iso_list, common_blocks, common_conn_map, added_blocks_list)

            # add neigbour nodes to the queues
            self._continue_bfs(pipelines_blocks, queue_list, conn_maps_blocks, added_blocks_list)

        # redefining input pipelines to use calculations
        # finding blocks in the common part
        # which can be used as the outputs
        common_outputs = dict()
        extra_blocks_list = [[] for _ in range(len(pipelines))]
        extra_inputs_list = [dict() for _ in range(len(pipelines))]
        extra_outputs_list = [dict() for _ in range(len(pipelines))]
        conns_to_remove_list = [[] for _ in range(len(pipelines))]
        conns_to_append_list = [[] for _ in range(len(pipelines))]
        pip_out_slots_list = []
        for pipeline in pipelines:
            output_slots = dict()
            for name, slot in pipeline.outputs.items():
                output_slots[slot] = name
            pip_out_slots_list.append(output_slots)

        for i in range(len(pipelines)):
            proceeded_blocks = set()
            for common_block in common_blocks:
                iso_block = block_iso_list[i][common_block]
                forward_blocks = conn_maps_blocks[i][iso_block]

                # seems like common part and
                # the pipeline are match
                if len(forward_blocks) == 0:
                    out_slot = next(iter(iso_block.slots.outputs.values()))
                    # need to add input plug and refresh
                    # output dictionary
                    if out_slot in pip_out_slots_list[i]:
                        input_plug = self._get_input_plug(out_slot)
                        extra_blocks_list[i].append(input_plug)
                        output_name = str(hash(common_block))
                        extra_inputs_list[i][output_name] = input_plug.get_single_input()
                        if output_name not in common_outputs:
                            common_outputs[output_name] = next(
                                iter(common_block.slots.outputs.values()))
                        extra_outputs_list[i][pip_out_slots_list[i][out_slot]] =\
                              next(iter(input_plug.slots.outputs.values()))
                    continue

                blocks_to_connect: List[BaseBlock] = []  # block from the orig pipeline
                for block in forward_blocks:
                    # if block is not presented in common part
                    if any([slot not in slot_iso_list[i] for slot in block.slots.inputs.values()]):
                        blocks_to_connect.append(block)

                # output is from common pipeline
                # input is from target
                for block in blocks_to_connect:
                    if block in proceeded_blocks:
                        continue
                    for inp_slot in block.slots.inputs.values():
                        corr_out = conn_maps_conns[i].get(inp_slot, None)
                        if corr_out is None:
                            continue
                        corr_out_cp = slot_iso_list[i].get(corr_out, None)
                        # found corresponding out slot in the common pipeline
                        # for the block (it's not necessary `common_block`)
                        if corr_out_cp is not None:
                            self._splice_pipelines(conns_to_append_list[i],
                                                   conns_to_remove_list[i],
                                                   inp_slot, common_outputs,
                                                   corr_out_cp, extra_inputs_list[i],
                                                   extra_blocks_list[i])
                    proceeded_blocks.add(block)

        self._optim_pipelines = []
        for i, pipeline in enumerate(pipelines):
            new_blocks = pipeline.nodes + extra_blocks_list[i]
            new_conns = []
            for conn in pipeline.connections:
                if conn.dst not in conns_to_remove_list[i]:
                    new_conns.append(conn)
            new_conns += conns_to_append_list[i]
            new_inputs = pipeline.inputs.copy()
            new_inputs.update(extra_inputs_list[i])
            new_outputs = pipeline.outputs.copy()
            new_outputs.update(extra_outputs_list[i])
            self._optim_pipelines.append(BasePipeline(new_blocks, new_conns,
                                            new_inputs, new_outputs))
        self._block_iso_list = block_iso_list
        self._new_blocks_list = extra_blocks_list
        common_inputs = self._get_common_input_dict(common_inputs_names, 
                                                    pipelines[0].inputs, slot_iso_list[0])
        common_conns = []
        for inp_slot, out_slot in common_conn_map.items():
            common_conns.append(Connection(out_slot, inp_slot))
        self._common_pipeline = BasePipeline(
            common_blocks, common_conns, common_inputs, common_outputs)
        self._set_random_state(self.random_state)

    @abstractmethod
    def get_score(self, data: Dict[str, BaseData], metrics: List[BaseMetric]) -> DataFrame:
        """Function to obtain results of different metrics for the models.
        """

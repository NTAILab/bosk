from abc import ABC
from typing import List
from bosk.block.base import BaseBlock
from bosk.block.zoo.input_plugs import InputBlock, TargetInputBlock
from bosk.pipeline.base import BasePipeline
from bosk.executor.topological import TopologicalExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.painter.topological import TopologicalPainter
from bosk.data import CPUData
from bosk.stages import Stage
from collections import deque, defaultdict
import joblib
from functools import cache
import numpy as np
from sklearn.model_selection import KFold
from typing import List, Callable


@cache
def get_block_hash(block: BaseBlock):
    return joblib.hash(block)


class BaseProfiler(ABC):
    def _get_aj_lists(self, pipeline):
        conn_map_blocks = defaultdict(set)  # output->input, forward pass
        conn_map_conns = dict()  # input->output, backward pass
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

    def __init__(self, models: List[BasePipeline], common_part: BasePipeline) -> None:
        """
        1. Step of bfs in common part is leading
        2. Each time when block is extracted on step 1, we look in the queue
        of other pipelines for isomorphic blocks (compare blocks + conns)
        3. If the isomorphic block is found for i'th pipeline, then we
        add his neighbour blocks in his queue
        4? If we didn't find the isomophic block, then we should raise the
        exception, because it can't be on another level
        5. Purpose of the algorithm - build two isomorphisms: for blocks and for
        connections. When the last block in common part's bfs is proceeded, we
        will be able to map common part's outputs and slots in other pipelines.
        """

        conn_map_blocks_cp, conn_map_conns_cp = self._get_aj_lists(common_part)
        begin_blocks_cp = [inp_conn.parent_block for inp_conn in common_part.inputs.values()]

        begin_blocks = []
        conn_maps_blocks = []
        conn_maps_conns = []
        visited = []
        queue = []
        block_iso_list = []
        conn_iso_list = []
        conns_to_remove_list = [set() for _ in range(len(models))]
        for pipeline in models:
            cur_blocks_al, cur_conns_al = self._get_aj_lists(pipeline)
            conn_maps_blocks.append(cur_blocks_al)
            conn_maps_conns.append(cur_conns_al)
            cur_begin_blocks = [inp_conn.parent_block for inp_conn in pipeline.inputs.values()]
            begin_blocks.append(cur_begin_blocks)
            visited.append(set(cur_begin_blocks))
            queue.append(deque(cur_begin_blocks))
            block_iso_list.append(dict())  # map from common part's blocks to model's
            conn_iso_list.append(dict())  # map from common part's conns to model's

        visited_cp = set(begin_blocks_cp)
        queue_cp = deque(begin_blocks_cp)
        while queue_cp:
            cur_block: BaseBlock = queue_cp.popleft()
            cur_block_hash = get_block_hash(cur_block)
            for i in range(len(models)):
                queue_hashes = [get_block_hash(block) for block in queue[i]]
                # find matching block in queue
                match_idxes = [i for i, hash in enumerate(queue_hashes) if hash == cur_block_hash]
                match_block = None
                # todo: need to optimize by adding input block in the optim_pipeline
                for idx in match_idxes:
                    if self._compare_blocks(cur_block, queue[i][idx], conn_map_conns_cp, conn_maps_conns[i], block_iso_list[i]):
                        match_block = queue[i][idx]
                        # delete matched block from the queue
                        del queue[i][idx]
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
                        queue[i].append(neig_node)

            # continue bfs for the common part
            for neig_node in conn_map_blocks_cp[cur_block]:
                if neig_node not in visited_cp:
                    visited_cp.add(neig_node)
                    queue_cp.append(neig_node)

        self.optim_pipelines: List[BasePipeline] = []
        for i, pipeline in enumerate(models):
            # All common part's inputs, must have isomorphic
            # slots in the pipeline and should NOT be excluded.
            new_inputs = pipeline.inputs.copy()
            for inp_name, inp_slot in common_part.inputs.items():
                iso_slot = conn_iso_list[i].get(inp_slot, None)
                assert iso_slot is not None, "Each input in the common part must have corresponding input in the pipeline"
                assert iso_slot == pipeline.inputs[inp_name], "Inputs in common part and pipeline must have the same name"
                # del new_inputs[inp_name]
            # The common part's output slot must have isomorphic
            # slot in the pipeline, for which we should make additional input.
            # Need to account that we may have common output
            #! just making new_inputs[out_name] = iso_slot should be enough
            #! it's not fair, validator and painter will raise exception
            # We also need to delete the original connection
            output_slots = dict()
            for name, slot in pipeline.outputs.items():
                output_slots[slot] = name
            new_outputs = pipeline.outputs.copy()
            # unable to make output slot directly as input,
            # so we need to add input blocks
            extra_blocks = []
            for out_name, out_slot in common_part.outputs.items():
                iso_slot = conn_iso_list[i].get(out_slot, None)
                assert iso_slot is not None, "Each output in the common part must have corresponding slot in the pipeline"
                if iso_slot in output_slots:
                    if iso_slot.meta.stages.transform:
                        input_plug = InputBlock()
                    else:
                        input_plug = TargetInputBlock()
                    extra_blocks.append(input_plug)
                    new_inputs[out_name] = next(iter(input_plug.slots.inputs.values()))
                    new_outputs[output_slots[iso_slot]] = next(iter(input_plug.slots.outputs.values()))
                    del output_slots[iso_slot]
                    continue
                # finding corresponding input slots
                k = 0
                # optimization for the slot removing process:
                # get the ordered dict for aj_list_conns
                # so we will know which index in the pipeline's conns list
                # should be removed
                for in_slot, corr_out_slot in conn_maps_conns[i].items():
                    if corr_out_slot == iso_slot:
                        new_inputs[f'{out_name}_{k}'] = in_slot
                        conns_to_remove_list[i].add(in_slot)
                        k += 1
            new_conns = []
            for conn in pipeline.connections:
                if conn.dst not in conns_to_remove_list[i]:
                    new_conns.append(conn)
            self.optim_pipelines.append(BasePipeline(pipeline.nodes + extra_blocks, new_conns, new_inputs, new_outputs))
            # todo: debug feature, remove the painting after
            # TopologicalPainter().from_pipeline(self.optim_pipelines[-1]).render(f'pipeline_{i}.png')
        self.common_pipeline = common_part
        # todo: debug feature, remove the painting after
        # TopologicalPainter().from_pipeline(common_part).render(f'common_part.png')

    def cv_score(self, data, n_folds, metrics: List[Callable]):
        n = None
        for value in data.values():
            if n is None:
                n = value.data.shape[0]
            else:
                assert value.data.shape[0] == n, "All inputs must have the same number of samples"
        idx = np.arange(n)
        k_fold = KFold(n_folds, shuffle=True)
        metrics_train_hist = []
        metrics_test_hist = []
        for i in range(len(self.optim_pipelines)):
            metrics_train_hist.append([[] for _ in range(len(metrics))])
            metrics_test_hist.append([[] for _ in range(len(metrics))])
        for i, (train_idx, test_idx) in enumerate(k_fold.split(idx)):
            print('Fold', i)

            train_dict = dict()
            test_dict = dict()
            for key, val in data.items():
                train_dict[key] = CPUData(val.data[train_idx])
                test_dict[key] = CPUData(val.data[test_idx])

            train_exec = TopologicalExecutor(self.common_pipeline, HandlingDescriptor.from_classes(Stage.FIT))
            common_train_res = train_exec(train_dict)
            test_exec = TopologicalExecutor(self.common_pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM))
            common_test_res = test_exec(test_dict)

            # todo: debug feature, remove the painting after
            if i == 0:
                TopologicalPainter().from_executor(train_exec).render('common_pipeline_fit.png')
                TopologicalPainter().from_executor(test_exec).render('common_pipeline_tf.png')

            for j, pip in enumerate(self.optim_pipelines):
                # build personal train dict
                cur_train_dict = dict()
                for key in pip.inputs.keys():
                    if key in train_dict:
                        cur_train_dict[key] = train_dict[key]
                    else:
                        # need to optimize by adding input block in the optim_pipeline
                        for common_out in common_train_res.keys():
                            if key.startswith(common_out):
                                cur_train_dict[key] = common_train_res[common_out]
                            else:
                                raise RuntimeError(f'Unable to find {common_out} in optim_pipeline')

                pip_tr_exec = TopologicalExecutor(pip, HandlingDescriptor.from_classes(Stage.FIT))
                pip_train_res = pip_tr_exec(cur_train_dict)

                # build personal test dict
                cur_test_dict = dict()
                for key in pip.inputs.keys():
                    if key in test_dict:
                        cur_test_dict[key] = test_dict[key]
                    else:
                        # need to optimize by adding input block in the optim_pipeline
                        for common_out in common_test_res.keys():
                            if key.startswith(common_out):
                                cur_test_dict[key] = common_test_res[common_out]
                            else:
                                raise RuntimeError(f'Unable to find {common_out} in optim_pipeline')

                pip_test_exec = TopologicalExecutor(pip, HandlingDescriptor.from_classes(Stage.TRANSFORM))
                pip_test_res = pip_test_exec(cur_test_dict)

                # todo: debug feature, remove the painting after
                if i == 0:
                    TopologicalPainter().from_executor(pip_tr_exec).render(f'optim_pipeline_{j}_fit.png')
                    TopologicalPainter().from_executor(pip_test_exec).render(f'optim_pipeline_{j}_tf.png')

                for k, metric in enumerate(metrics):
                    metric_train_res = metric(train_dict, pip_train_res)
                    metrics_train_hist[j][k].append(metric_train_res)
                    print(f'\tTrain fold res (model {j} metric {k}):', metric_train_res)
                    metric_test_res = metric(test_dict, pip_test_res)
                    metrics_test_hist[j][k].append(metric_test_res)
                    print(f'\tTest fold res (model {j} metric {k}):', metric_test_res)

        print(f'Average results for {n_folds} folds:')
        for i in range(len(self.optim_pipelines)):
            print(f'Model {i}:')
            for j in range(len(metrics)):
                print(f'\tMetric {j} train:', np.mean(metrics_train_hist[i][j]))
                print(f'\tMetric {j} test:', np.mean(metrics_test_hist[i][j]))

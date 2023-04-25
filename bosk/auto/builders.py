from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional, Type

from bosk.block.base import BlockInputData

from ..executor.base import BaseExecutor
from ..pipeline.base import BasePipeline
from ..pipeline.connection import Connection
from ..stages import Stage
from ..data import BaseData
from .growing_strategies import GrowingStrategy
from .layers import Layer


class SequentialPipelineBuilder:
    base_input_names = ['X', 'y']
    pipelines: List[BasePipeline]
    history: Mapping[str, List[float]]

    def __init__(self, executor_cls: Type[BaseExecutor],
                 growing_strategy: GrowingStrategy,
                 **inputs: BaseData):
        super().__init__()
        self.executor_cls = executor_cls
        self.inputs = inputs
        self.prev_step_inputs: Optional[BlockInputData] = None
        self.pipelines = []
        self.growing_strategy = growing_strategy
        self.__growing_state: Dict[str, Any] = dict()
        self.history = defaultdict(list)

    def append(self, layer: Layer) -> bool:
        if self.prev_step_inputs is None:
            step_inputs = {k: v for k, v in self.inputs.items() if k in layer.inputs}
        else:
            prev_pipeline = self.pipelines[-1]
            available_outputs = [k for k in layer.inputs if k in prev_pipeline.outputs]
            prev_transformer = self.executor_cls(prev_pipeline, Stage.TRANSFORM, outputs=available_outputs)
            step_inputs = prev_transformer(self.prev_step_inputs)
        # append static inputs (e.g. 'y')
        step_inputs = {**step_inputs}
        step_inputs.update({
            k: self.inputs[k]
            for k in layer.inputs
            if k not in step_inputs and k in self.inputs
        })
        pipeline, metrics = layer.fit(step_inputs)
        self.pipelines.append(pipeline)
        self.prev_step_inputs = step_inputs
        if not self.growing_strategy.need_grow(pipeline, metrics, self.executor_cls, self.__growing_state):
            self.pipelines = self.growing_strategy.trim(self.pipelines, self.__growing_state)
            return False
        self._log_metrics(metrics)
        return True

    def _log_metrics(self, metrics):
        if metrics is None:
            return
        for k, v in metrics.items():
            self.history[k].append(v)

    def build(self, map_inputs: Mapping[str, str]):
        """Build the pipeline.

        Args:
            map_inputs: Mapping of inputs that are not given at a TRANSFORM stage.
                        For example, if layers depend on `X` and `X_original`,
                        the pipeline will have only `X` input, and the mapping should be
                        `{"X_original": "X"}`.

        Returns:
            Pipeline.

        """
        nodes = []
        connections = []
        outputs = self.pipelines[-1].outputs
        inputs = {
            k: v
            for k, v in self.pipelines[0].inputs.items()
            if k not in map_inputs  # consider only inputs that will be given
        }
        for alias_name, given_input_name in map_inputs.items():
            given_block = self.pipelines[0].inputs[given_input_name].parent_block
            given_output_slot = given_block.slots.outputs[given_block.default_output]
            connections.append(
                Connection(
                    given_output_slot,
                    self.pipelines[0].inputs[alias_name]
                )
            )

        # iterate over layers
        for i, pipeline in enumerate(self.pipelines):
            nodes.extend(pipeline.nodes)
            connections.extend(pipeline.connections)
            if i != 0:
                # connect to the previous pipeline outputs
                prev_pipeline = self.pipelines[i - 1]
                fulfilled_inputs = set()
                for out_name, out_slot in prev_pipeline.outputs.items():
                    if out_name in pipeline.inputs:
                        # matching slots by names
                        connections.append(Connection(out_slot, pipeline.inputs[out_name]))
                        fulfilled_inputs.add(out_name)
                # connect the base input slots to the layer, if they are not passed
                for base_inp in self.base_input_names:
                    if base_inp in pipeline.inputs and base_inp not in fulfilled_inputs:
                        target_input_block = self.pipelines[0].inputs[base_inp].parent_block
                        target_output_slot = target_input_block.slots.outputs[target_input_block.default_output]
                        connections.append(Connection(target_output_slot, pipeline.inputs[base_inp]))

        return BasePipeline(
            nodes=nodes,
            connections=connections,
            inputs=inputs,
            outputs=outputs,
        )


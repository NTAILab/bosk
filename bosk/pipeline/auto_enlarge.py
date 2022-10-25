from abc import ABC, abstractmethod
from typing import Mapping, List
from ..data import Data
from ..block import BaseBlock
from .base import BasePipeline, BlockInputSlot, BlockOutputSlot

class EnlargingMonitor(ABC):
    """
    Class for monitoring pipeline enlarging process.

    """

    @abstractmethod
    def f_need_to_enlarge(self, output_data: Mapping[str, Data]) -> bool:
        """
        Method that tells if there is a need to enlarge the pipeline. Should be called 
        on each iteration of the pipeline building.
        
        Args:
            output_data: Mapping from output slots name to results, which 
            should contain metrics you interested in.
        
        Returns:
            Boolean indicating if there is a need to enlarge the pipeline.

        """
    
    @abstractmethod
    def best_iteration(self) -> int:
        """
        Method that returns number of the best iteration.

        Returns:
            Number of the best iteration.
        """

class EnlargingPipeline(ABC):
    """
    Abstract class that defines a pipeline with the enlarging ability.
    """
    def __init__(self, enlarging_monitor: EnlargingMonitor):
        self.enl_monitor = enlarging_monitor
    
    @abstractmethod
    def enlarge(self):
        """
        Method that provides enlarging mechanism.
        """

    @abstractmethod
    def restore_state(self, iter_num: int):
        """
        Method that restores pipeline state at the moment of the enlargement iteration.

        Args:
            iter_num: Number of the enlargement iteration to restore at.
        """

# abstractness should be discussed
class ExampleEnlPipeLine(EnlargingPipeline):
    def __process_init_pipeline(self, pipeline: BasePipeline, inputs: List[BlockInputSlot], outputs: List[BlockOutputSlot]):
        conn_dict = dict()
        for conn in pipeline.connections:
            assert(conn.dst not in conn_dict), f"Input slot {conn.dst.name} (id {hash(conn.dst)}) is used more than once"
            conn_dict[conn.dst] = conn.src
        self.conn_dict = conn_dict
        self.inputs = inputs
        self.outputs = outputs

    # metrics as functions are better? how to connect metric blocks? -> no metrics, all in the monitor
    # include inputs and outputs in the pipeline class?
    def __init__(self, init_pipeline: BasePipeline, inputs: List[BlockInputSlot], outputs: List[BlockOutputSlot], enlarging_monitor: EnlargingMonitor):
        self.enlarging_monitor = enlarging_monitor
        super().__init__(enlarging_monitor)
        self.__process_init_pipeline(init_pipeline, inputs, outputs)


    def fit(self, input_values: Mapping[str, Data]):
        passed = 0
        for inp_name, inp_slot in self.inputs.items():
            # assert(inp_slot in conn_dict), f"Unable to find block with input {inp_name} (id {hash(inp_slot)}) in pipeline"
            assert(inp_name in input_values), f"Unable to find input slot {inp_name} (id {hash(inp_slot)}) in input data"
            passed += 1
        assert(passed == len(input_values)), f"Input values are incompatible with pipeline input slots"




from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.descriptor import HandlingDescriptor
from bosk.executor.base import BaseExecutor
from bosk.stages import Stage
from bosk.data import Data
from bosk.block.base import BaseBlock
from typing import Dict, Type, Optional, Sequence, Tuple, List, Any, Set, TypeVar

T = TypeVar('T')


def fit_pipeline(pipeline: BasePipeline, data: Dict[str, Data], exec_cls: Type[BaseExecutor],
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None) -> Tuple[BasePipeline, Dict[str, Data]]:
    executor = exec_cls(pipeline, HandlingDescriptor.from_classes(Stage.FIT), inputs, outputs)
    fit_output = executor(data)
    return executor.pipeline, fit_output


def get_all_subclasses(cls: T) -> Set[T]:
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])


def connect_chain(chain: List[BaseBlock],
                  out_name: str = 'output', in_name: str = 'X') -> List[Connection]:
    return [Connection(chain[i - 1].slots.outputs[out_name],
                       chain[i].slots.inputs[in_name]) for i in range(1, len(chain))]

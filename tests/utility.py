"""Script containing useful procedures for the `bosk.tests` package."""

from bosk.pipeline.base import BasePipeline, Connection
from bosk.executor.base import BaseExecutor
from bosk.stages import Stage
from bosk.data import Data
from bosk.block.base import BaseBlock
from typing import Dict, Type, Optional, Sequence, Tuple, List, Set, TypeVar
import os
import logging

T = TypeVar('T')


def fit_pipeline(pipeline: BasePipeline, data: Dict[str, Data], exec_cls: Type[BaseExecutor],
                 inputs: Optional[Sequence[str]] = None,
                 outputs: Optional[Sequence[str]] = None) -> Tuple[BasePipeline, Dict[str, Data]]:
    """Function that fits the pipeline with the preferred executor. Returns fitted pipeline and the
    pipeline's output. Is needed to avoid the boilerplate code.
    """
    executor = exec_cls(pipeline, Stage.FIT, inputs, outputs)
    fit_output = executor(data)
    return executor.pipeline, fit_output


def get_all_subclasses(cls: T) -> Set[T]:
    """Function that retrieves all the inheritors of some basic class.
    Is needed as the `__subclasses__` method returns only first generation
    inheritors.
    """
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)])


def connect_chain(chain: List[BaseBlock],
                  out_name: str = 'output', in_name: str = 'X') -> List[Connection]:
    """Function that makes connected computational graph's branch out of
    blocks list. Is needed to avoid the boilerplate code.
    """
    return [Connection(chain[i - 1].slots.outputs[out_name],
                       chain[i].slots.inputs[in_name]) for i in range(1, len(chain))]


def log_test_name() -> None:
    logging.info('Starting the "%s" test', os.environ.get('PYTEST_CURRENT_TEST').split(' ', 1)[0])

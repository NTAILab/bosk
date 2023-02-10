from bosk.pipeline.base import BasePipeline
from bosk.executor.descriptor import HandlingDescriptor
from bosk.executor.base import BaseExecutor
from bosk.stages import Stage
from bosk.data import Data
from typing import Dict, Type, Optional, Sequence, Tuple

def fit_pipeline(pipeline: BasePipeline, data: Dict[str, Data], exec_cls: Type[BaseExecutor],
                inputs: Optional[Sequence[str]] = None,
                outputs: Optional[Sequence[str]] = None) -> Tuple[BasePipeline, Dict[str, Data]]:
    executor = exec_cls(pipeline, HandlingDescriptor.from_classes(Stage.FIT), inputs, outputs)
    fit_output = executor(data)
    return executor.pipeline, fit_output

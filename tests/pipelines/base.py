from abc import ABC, abstractmethod
from typing import Dict
from bosk.pipeline.base import BasePipeline
from bosk.data import Data
from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage

class PipelineTestBase(ABC):

    def fit_test(self):
        pipeline = self.get_pipeline()
        data = self.get_fit_data()
        executor = RecursiveExecutor(pipeline, HandlingDescriptor.from_classes(Stage.FIT))
        executor(data)

    @abstractmethod
    def get_pipeline(self) -> BasePipeline:
        """Method for the BasePipeline retreiving."""
    
    @abstractmethod
    def get_fit_data(self) -> Dict[str, Data]:
        """Method for retreiving the data to fit the pipeline."""
    
    @abstractmethod
    def get_transform_data(self) -> Dict[str, Data]:
        """Method for retreiving the data to execute 
            transform stage of the pipeline.
        """    


from abc import ABC, abstractmethod
from typing import Dict
from bosk.pipeline.base import BasePipeline
from bosk.data import Data


class BasePipelineGetter(ABC):
    """Abstract class for getting the test pipeline."""
    def __init__(self) -> None:
        super().__init__()
    
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

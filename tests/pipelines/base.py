from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple
from bosk.pipeline.base import BasePipeline
from bosk.data import Data
from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.stages import Stage
from ..utility import fit_pipeline
import logging


class BasePipelineTest(ABC):

    def fit_test(self):
        _, fit_output = fit_pipeline(self.get_pipeline(), self.get_fit_data(),
                                     RecursiveExecutor, *self.get_fit_in_out())
        logging.info('Test "%s" provided following outputs throung the fit procedure: %r',
                     self.__class__.__name__, list(fit_output.keys()))

    def transform_test(self):
        fitted_pipeline, _ = fit_pipeline(
            self.get_pipeline(), self.get_fit_data(), RecursiveExecutor, *self.get_fit_in_out())
        data = self.get_transform_data()
        executor = RecursiveExecutor(fitted_pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                                    *self.get_transform_in_out())
        tf_output = executor(data)
        logging.info('Test "%s" provided following outputs throung the transform procedure: %r',
                     self.__class__.__name__, list(tf_output.keys()))

    @abstractmethod
    def get_pipeline(self) -> BasePipeline:
        """Method for the BasePipeline retreiving."""

    @abstractmethod
    def get_fit_data(self) -> Dict[str, Data]:
        """Method for retreiving the data to fit the pipeline."""

    def get_fit_in_out(self) -> Tuple[Optional[Sequence[str]], Optional[Sequence[str]]]:
        """Method for retreiving the inputs and outputs names to fit the pipeline."""
        return None, None

    @abstractmethod
    def get_transform_data(self) -> Dict[str, Data]:
        """Method for retreiving the data to execute the
            transform stage of the pipeline.
        """

    def get_transform_in_out(self) -> Tuple[Optional[Sequence[str]], Optional[Sequence[str]]]:
        """Method for retreiving the inputs and outputs names to to execute the
            transform stage of the pipeline."""
        return None, None

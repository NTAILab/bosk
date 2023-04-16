"""Script that contains the common base for the different pipelines tests."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Sequence, Tuple
from bosk.pipeline.base import BasePipeline
from bosk.data import Data
from bosk.executor.recursive import RecursiveExecutor
from bosk.stages import Stage
from ..utility import fit_pipeline, log_test_name
import logging


class BasePipelineTest(ABC):
    """Base class to perform test of the particular pipeline. It is needed
    to check work of the different blocks of the `bosk.block.zoo` package."""

    random_seed = 42

    def fit_test(self):
        """Test of the fit stage."""
        log_test_name()
        _, fit_output = fit_pipeline(self.get_pipeline(), self.get_fit_data(),
                                     RecursiveExecutor, *self.get_fit_in_out())
        logging.info('Test "%s" provided following outputs throung the fit procedure: %r',
                     self.__class__.__name__, list(fit_output.keys()))

    def transform_test(self):
        """Test of the transform stage."""
        log_test_name()
        fitted_pipeline, _ = fit_pipeline(
            self.get_pipeline(), self.get_fit_data(), RecursiveExecutor, *self.get_fit_in_out())
        data = self.get_transform_data()
        executor = RecursiveExecutor(fitted_pipeline, Stage.TRANSFORM, *self.get_transform_in_out())
        tf_output = executor(data)
        logging.info('Test "%s" provided following outputs throung the transform procedure: %r',
                     self.__class__.__name__, list(tf_output.keys()))

    @abstractmethod
    def _get_pipeline(self) -> BasePipeline:
        """Method for the BasePipeline building by a child class."""

    def get_pipeline(self) -> BasePipeline:
        """Method for the BasePipeline retreiving."""
        pipeline = self._get_pipeline()
        pipeline.set_random_state(self.random_seed)
        return pipeline

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

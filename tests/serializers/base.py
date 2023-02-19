from abc import ABC, abstractmethod
from bosk.pipeline.serializer.base import BasePipelineSerializer
from bosk.executor.recursive import RecursiveExecutor
from bosk.executor.descriptor import HandlingDescriptor
from bosk.pipeline.base import BasePipeline
from bosk.stages import Stage
import numpy as np
from ..pipelines import *
from ..pipelines.base import BasePipelineTest as BPT
from ..utility import get_all_subclasses, fit_pipeline
from . import TMP_SAVE_DIR
from typing import List
from os import remove
import logging


class BaseSerializerTest(ABC):

    @abstractmethod
    def get_serializers(self) -> List[BasePipelineSerializer]:
        ...

    def _check_inputs_outputs(self, unserialized_pipeline: BasePipeline):
        assert len(self.pip_inputs) == len(unserialized_pipeline.inputs),\
            'The unserialized and the original pipelines have different numbers of inputs'
        for inp in self.pip_inputs:
            assert inp in unserialized_pipeline.inputs,\
                f"The unserialized pipeline doesn't have '{inp}' input"
            assert self.pip_inputs[inp].parent_block.__class__ == \
                unserialized_pipeline.inputs[inp].parent_block.__class__,\
                f"Input '{inp}' points to different blocks"
        assert len(self.pip_outputs) == len(unserialized_pipeline.outputs),\
            'The unserialized and the original pipelines have different numbers of outputs'
        for out in self.pip_outputs:
            assert out in unserialized_pipeline.outputs,\
                f"The unserialized pipeline doesn't have '{out}' output"
            assert self.pip_outputs[out].parent_block.__class__ == \
                unserialized_pipeline.outputs[out].parent_block.__class__,\
                f"Output '{out}' heads from different blocks"

    def _check_outputs(self, true_outs, pred_outs):
        eps = 1e-15
        assert len(true_outs) == len(pred_outs),\
            'Number of true outputs is different from the ones from the deserialized pipeline'
        for out in true_outs:
            assert out in pred_outs,\
                f"There is no output '{out}' in the result of the deserialized pipeline computation"
            assert np.sum(np.abs(pred_outs[out] - true_outs[out])) < eps,\
                f"The output '{out}' is different from the reference value"

    def serializer_test(self):
        pip_wrappers = get_all_subclasses(BPT)
        logging.info('Following pipeline wrappers were found: %r',
                     [p.__name__ for p in pip_wrappers])
        for pw_cls in pip_wrappers:
            logging.info('Begin test for %s pipeline', pw_cls.__name__)
            pw = pw_cls()
            pipeline = pw.get_pipeline()
            self.pip_inputs, self.pip_outputs = pipeline.inputs, pipeline.outputs
            fitted_pipeline, fit_output = fit_pipeline(pipeline, pw.get_fit_data(),
                                                       RecursiveExecutor, *pw.get_fit_in_out())
            tf_data = pw.get_transform_data()
            executor = RecursiveExecutor(fitted_pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                                         *pw.get_transform_in_out())
            tf_output = executor(tf_data)
            ser_list = self.get_serializers()
            logging.info('%s has provided %i variety(ies) of the serializer',
                         self.__class__.__name__, len(ser_list))
            for i, serializer in enumerate(ser_list):
                logging.info('Test of the serializer var. #%i', i + 1)
                pipeline = pw.get_pipeline()
                unfit_filename = TMP_SAVE_DIR + \
                    f'/{self.__class__.__name__}_{i + 1}_{pw_cls.__name__}_unfit.tmp'
                serializer.dump(pipeline, unfit_filename)
                logging.info('Dump of the unfitted pipeline is done')
                pipeline = serializer.load(unfit_filename)
                self._check_inputs_outputs(pipeline)
                fitted_pipeline, cur_fit_output = fit_pipeline(pipeline, pw.get_fit_data(),
                                                               RecursiveExecutor, *pw.get_fit_in_out())
                self._check_outputs(fit_output, cur_fit_output)
                fit_filename = TMP_SAVE_DIR + \
                    f'/{self.__class__.__name__}_{i + 1}_{pw_cls.__name__}_fit.tmp'
                serializer.dump(fitted_pipeline, fit_filename)
                logging.info('Dump of the fitted pipeline is done')
                fitted_pipeline = serializer.load(fit_filename)
                self._check_inputs_outputs(fitted_pipeline)
                executor = RecursiveExecutor(fitted_pipeline, HandlingDescriptor.from_classes(Stage.TRANSFORM),
                                             *pw.get_transform_in_out())
                cur_tf_output = executor(tf_data)
                self._check_outputs(tf_output, cur_tf_output)
                logging.info('Test is successful, deleting temp files')
                remove(unfit_filename)
                remove(fit_filename)

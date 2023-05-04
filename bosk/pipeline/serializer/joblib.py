"""Joblib-based serialization.
"""
from .base import BaseBlockSerializer, BasePipelineSerializer, BasePipeline
from ...block.base import BaseBlock
from joblib import load as joblib_load, dump as joblib_dump
from typing import Any, Optional


class JoblibBlockSerializer(BaseBlockSerializer):
    """Joblib-based block serialization.

    Args:
        compress: Compression level (see :py:func:`joblib.dump`).
        protocol: Pickle protocol (see :py:func:`joblib.dump`).

    """
    def __init__(self, compress: int = 0, protocol: Optional[Any] = None):
        self.compress = compress
        self.protocol = protocol

    def dump(self, block: BaseBlock, out_file):
        joblib_dump(
            block,
            out_file,
            compress=self.compress,
            protocol=self.protocol
        )

    def load(self, in_file) -> BaseBlock:
        block = joblib_load(in_file)
        assert isinstance(block, BaseBlock), "Loaded object should be a block"
        return block


class JoblibPipelineSerializer(BasePipelineSerializer):
    """Joblib-based pipeline serialization.

    Consider using :py:class:`bosk.pipeline.serializer.ZipPipelineSerializer` instead
    if there will be any issues.

    Args:
        compress: Compression level (see :py:func:`joblib.dump`).
        protocol: Pickle protocol (see :py:func:`joblib.dump`).

    """
    def __init__(self, compress: int = 0, protocol: Optional[Any] = None):
        self.compress = compress
        self.protocol = protocol

    def dump(self, pipeline: BasePipeline, out_file):
        joblib_dump(
            pipeline,
            out_file,
            compress=self.compress,
            protocol=self.protocol
        )

    def load(self, in_file) -> BasePipeline:
        pipeline = joblib_load(in_file)
        assert isinstance(pipeline, BasePipeline), "Loaded object should be a pipeline"
        return pipeline

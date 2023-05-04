"""skops-based serialization.
"""
from .base import BasePipelineSerializer, BaseBlockSerializer, BasePipeline
from ...block.base import BaseBlock
from skops.io import load as skops_load, dump as skops_dump
from typing import Any, Optional


class SkopsBlockSerializer(BaseBlockSerializer):
    """Skops-based pipeline serialization.

    Skops is more secure than Joblib, but doesn't allow to serialize
    the whole pipelines.
    If you are going to do so, consider :py:class:`bosk.pipeline.serializer.ZipPipelineSerializer`.

    Args:
        trusted: Will the given data be trusted or not.
                 Note that it is True by default.

    """
    def __init__(self, trusted: bool = True):
        self.trusted = trusted

    def dump(self, block: BaseBlock, out_file):
        skops_dump(
            block,
            out_file,
        )

    def load(self, in_file) -> BaseBlock:
        block = skops_load(in_file, trusted=self.trusted)
        assert isinstance(block, BaseBlock), "Loaded object should be a block"
        return block

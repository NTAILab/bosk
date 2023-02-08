"""skops-based serialization.
"""
from .base import BaseSerializer, BasePipeline
from skops.io import load as skops_load, dump as skops_dump
from typing import Any, Optional


class SkopsSerializer(BaseSerializer):
    def dump(self, pipeline: BasePipeline, out_file):
        skops_dump(
            pipeline,
            out_file,
        )

    def load(self, in_file) -> BasePipeline:
        pipeline = skops_load(in_file)
        assert isinstance(pipeline, BasePipeline), "Loaded object should be a pipeline"
        return pipeline

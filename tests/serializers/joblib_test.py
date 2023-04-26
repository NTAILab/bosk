from .base import BaseSerializerTest as BST
from bosk.pipeline.serializer.base import BasePipelineSerializer
from bosk.pipeline.serializer.joblib import JoblibPipelineSerializer
from typing import List


class JoblibSerializerTest(BST):
    def get_serializers(self) -> List[BasePipelineSerializer]:
        return [JoblibPipelineSerializer()]

from .base import BaseSerializerTest as BST
from bosk.pipeline.serializer.base import BasePipelineSerializer
from bosk.pipeline.serializer.zip import ZipPipelineSerializer
from bosk.pipeline.serializer.joblib import JoblibBlockSerializer
from bosk.pipeline.serializer.skops import SkopsBlockSerializer
from typing import List


class ZipSerializerTest(BST):
    def get_serializers(self) -> List[BasePipelineSerializer]:
        return [ZipPipelineSerializer(JoblibBlockSerializer()), ZipPipelineSerializer(SkopsBlockSerializer())]

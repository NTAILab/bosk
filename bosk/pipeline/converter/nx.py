from functools import singledispatchmethod
import networkx as nx
from ...visitor.base import BaseVisitor
from ..base import BasePipeline
from ...block.base import BaseBlock
from ..connection import Connection


class NetworkXConverter:

    class Visitor(BaseVisitor):
        def __init__(self, graph: nx.Graph):
            self.graph = graph

        @singledispatchmethod
        def visit(self, obj):
            pass  # ignore extra entities

        @visit.register
        def _(self, block: BaseBlock):
            self.graph.add_node(block)

        @visit.register
        def _(self, connection: Connection):
            self.graph.add_edge(connection.src.parent_block, connection.dst.parent_block)

    def __init__(self):
        self.graph = nx.DiGraph()
        self.visitor = self.Visitor(self.graph)

    def __call__(self, pipeline: BasePipeline) -> nx.Graph:
        pipeline.accept(self.visitor)
        return self.graph

from collections import defaultdict
from typing import Optional, Sequence
from ..block.base import BaseBlock, BlockInputSlot
from ..pipeline.connection import Connection
from .base import BasePainter
from ..pipeline.base import BasePipeline
from ..executor.base import BaseExecutor
import graphviz as gv


class GraphvizPainter(BasePainter):
    """Painter that uses Graphviz library to draw the computational graph of a pipeline.

    Attributes:
        levels_sep: The painter's parameter, which determines the distance between the computational graph's levels.
            See http://graphviz.org/docs/attrs/ranksep/.
        dpi: The dpi of the output computational graph images, formated in raster graphics (.png, .jpeg, etc.).
        rankdir: The direction of the computational graph edges. See https://graphviz.org/docs/attrs/rankdir/.

    Args:
        graph_levels_sep: Sets :attr:`levels_sep`.
        figure_dpi: Sets :attr:`dpi`.
        figure_rankdir: Sets :attr:`rankdir`.
    """

    def __init__(self, graph_levels_sep: float = 1.0, figure_dpi: int = 150, figure_rankdir: str = 'LR',
                 render_groups: bool = True):
        self._levels_sep = graph_levels_sep
        self._dpi = figure_dpi
        self._rankdir = figure_rankdir
        self._render_groups = render_groups
        self._graph = gv.Digraph('DeepForestGraph', renderer='cairo',
                                 formatter='cairo', node_attr={'shape': 'record'})
        self._f_used = False

    def _add_node(self, block: BaseBlock, style: str = 'solid', color: str = 'black') -> None:
        """Method that adds a node to the graph.
        """
        inputs = block.slots.inputs
        outputs = block.slots.outputs
        inputs_info = '|'.join([f'<i{hash(slot)}> {name}' for name, slot in inputs.items()])
        outputs_info = '|'.join([f'<o{hash(slot)}> {name}' for name, slot in outputs.items()])
        self._graph.node(f'block{id(block)}', f'{repr(block)}|{{{{{inputs_info}}}|{{{outputs_info}}}}}',
                         style=style, color=color)

    def _add_edge(self, connection: Connection, style: str = 'solid', color: str = 'black') -> None:
        """Method that adds an edge to the graph.
        """
        self._graph.edge(f'block{id(connection.src.parent_block)}:o{hash(connection.src)}',
                         f'block{id(connection.dst.parent_block)}:i{hash(connection.dst)}',
                         style=style, color=color)

    def _add_input(self, name: str, input_slot: BlockInputSlot, style: str = 'solid', color: str = 'red') -> None:
        """Method that adds pipeline's input to the graph.
        """
        inp_hash = hash(name)
        self._graph.node(f'inp_{inp_hash}', f'<I_{inp_hash}> {name}', style=style, color=color)
        self._graph.edge(f'inp_{inp_hash}:I_{inp_hash}', f'block{id(input_slot.parent_block)}:i{hash(input_slot)}',
                         style=style, color=color)

    def _add_output(self, name: str, output_slot: BlockInputSlot, style: str = 'solid', color: str = 'red') -> None:
        """Method that adds pipeline's output to the graph.
        """
        out_hash = hash(name)
        self._graph.node(f'out_{out_hash}', f'<O_{out_hash}> {name}', style=style, color=color)
        self._graph.edge(f'block{id(output_slot.parent_block)}:o{hash(output_slot)}', f'out_{out_hash}:O_{out_hash}',
                         style=style, color=color)

    def from_pipeline(self, pipeline: BasePipeline) -> BasePainter:
        """Method that parses a pipeline and make internal representation
        of the computational graph to render its image in the :meth:`render` method.
        """
        assert not self._f_used, "You've already built the graph"
        for block in pipeline.nodes:
            self._add_node(block)
        if self._render_groups:
            groups = defaultdict(list)
            for block in pipeline.nodes:
                for g in block.slots.groups:
                    groups[g].append(block)
            for i, (group, group_nodes) in enumerate(groups.items()):
                with self._graph.subgraph(name=f'cluster_{i}') as c:
                    c.attr(style='filled', color='lightgrey')
                    c.node_attr.update(style='filled', color='white')
                    c.attr(label=repr(group))
                    for block in group_nodes:
                        c.node(f'block{id(block)}')

        for conn in pipeline.connections:
            self._add_edge(conn)
        for inp_name, inp_slot in pipeline.inputs.items():
            self._add_input(f'Input "{inp_name}"', inp_slot)
        for out_name, out_slot in pipeline.outputs.items():
            self._add_output(f'Output "{out_name}"', out_slot)
        self._f_used = True
        return self

    def from_executor(self, executor: BaseExecutor) -> BasePainter:
        raise NotImplementedError()

    def render(self, output_filename: str, format: Optional[str] = None) -> None:
        """Method that renders the computational graph's image.
        """
        assert self._f_used, "You must build the graph firstly. Use 'from_pipeline' or 'from_executor' methods"
        assert format is None or format in gv.FORMATS, f"Unable to render graph into {format} format"
        if format is not None:
            output_filename += f'.{format}'
        self._graph.attr(rankdir=self._rankdir, ranksep=str(self._levels_sep), dpi=str(self._dpi))
        self._graph.render(outfile=output_filename, cleanup=True)

    def available_formats(self) -> Sequence[str]:
        return gv.FORMATS

    @property
    def rankdir(self) -> str:
        """The direction of the computational graph edges. See https://graphviz.org/docs/attrs/rankdir/.
        """
        return self._rankdir

    @rankdir.setter
    def rankdir(self, value: str) -> None:
        self._rankdir = value

    @property
    def levels_sep(self) -> float:
        """The painter's parameter, which determines the distance between the computational graph's levels.
        See http://graphviz.org/docs/attrs/ranksep/.
        """
        return self._levels_sep

    @levels_sep.setter
    def levels_sep(self, value) -> None:
        self._levels_sep = value

    @property
    def dpi(self) -> int:
        """The dpi of the output computational graph images, formated in raster graphics (.png, .jpeg, etc.).
        """
        return self._dpi

    @dpi.setter
    def dpi(self, value) -> None:
        self._dpi = value

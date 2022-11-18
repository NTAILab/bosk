from .base import BasePainter
from .graphviz import GraphvizPainter
from ..executor.topological import TopologicalExecutor
from ..executor.base import BaseExecutor

class TopologicalPainter(GraphvizPainter):
    """Painter that performes the computational graph drawing in accordance with :class:`TopologicalExecutor`.
    Based on the :class:`GraphvizPainter`.

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
    def __init__(self, graph_levels_sep: float = 1, figure_dpi: int = 150, figure_rankdir: str = 'LR'):
        super().__init__(graph_levels_sep, figure_dpi, figure_rankdir)
    
    def from_executor(self, executor: BaseExecutor) -> BasePainter:
        """Method that parses a :class:`TopologicalExecutor` and make internal representation
        of the computational graph to render its image in the :meth:`render` method.

        * The solid black edges and nodes are the ones that will be used during calculations. 
        * The dashed will be skipped during the optimization. 
        * The red colored nodes mean used inputs and outputs, the red colored edges signalize that
          type of the block's input slot was misspecified: according to the :attr:`stage` metainformation 
          the connection should be used, but the corresponding block won't be used because of the optimization.
        * The blue colored nodes mean skipped inputs and outputs. They were specified in the pipeline,
            but not in the executor.
        """
        assert not self._f_used, "You've already built the graph"
        assert isinstance(executor, TopologicalExecutor), \
            f"This painter works only with topological executor, got {executor.__class__.__name__}"
        
        if executor.outputs is None:
            output_blocks = [slot.parent_block for slot in executor.pipeline.outputs.values()]
        else:
            output_blocks = [executor.pipeline.outputs[slot_name].parent_block for slot_name in executor.outputs]
        backward_pass = executor._dfs(executor._get_backward_aj_list(), output_blocks)

        if executor.inputs is None:
            input_blocks = [slot.parent_block for slot in executor.pipeline.inputs.values()]
        else:
            input_blocks = [executor.pipeline.inputs[slot_name].parent_block for slot_name in executor.inputs]
        forward_pass = executor._dfs(executor._get_forward_aj_list(backward_pass), input_blocks)
        used_blocks = backward_pass & forward_pass

        for block in executor.pipeline.nodes:
            node_style = 'dashed' if block not in used_blocks else ''
            self._add_node(block, node_style)

        for conn in executor.pipeline.connections:
            is_conn_req = executor.slots_handler.is_slot_required(conn.dst)
            edge_style = 'dashed' if not is_conn_req else ''
            edge_color = 'red' if conn.dst.parent_block not in used_blocks and is_conn_req else 'black'
            self._add_edge(conn, edge_style, edge_color)

        for inp_name, inp_slot in executor.pipeline.inputs.items():
            node_style = 'solid' if executor.slots_handler.is_slot_required(inp_slot) else 'dashed'
            if executor.inputs is None or inp_name in executor.inputs:
                node_color = 'red'
            else:
                node_color = 'blue'
            self._add_input(f'Input "{inp_name}"', inp_slot, node_style, node_color)
        
        for out_name, out_slot in executor.pipeline.outputs.items():
            node_style = 'solid' if out_slot.parent_block in used_blocks else 'dashed'
            if executor.outputs is None or out_name in executor.outputs:
                node_color = 'red'
            else:
                node_color = 'blue'
            self._add_output(f'Output "{out_name}"', out_slot, node_style, node_color)
        
        self._f_used = True
        return self
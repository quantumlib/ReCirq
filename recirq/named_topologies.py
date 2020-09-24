import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

import cirq
import re


def draw_gridlike(graph, highlight=None, **kwargs):
    pos = {}
    for node in graph.nodes:
        if isinstance(node, cirq.GridQubit):
            pos[node] = node.col, node.row
        elif isinstance(node, tuple):
            row, col = node
            pos[node] = col, row
        else:
            raise ValueError("Don't know how to position {}".format(node))

    if highlight is not None:
        colors = ['grey' if n not in highlight else 'red' for n in graph.nodes]
    else:
        colors = None

    nx.draw_networkx(graph, pos=pos, node_color=colors, **kwargs)


class LineTopology:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.name = f'{self.n_qubits}q-line'
        self.graph = nx.from_edgelist([(i1, i2) for i1, i2
                                       in zip(range(self.n_qubits), range(1, self.n_qubits))])

    @classmethod
    def from_name(cls, topology_name):
        ma = re.match(r'(\d+)q-line', topology_name)
        if ma is None:
            raise ValueError("Could not parse topology name: {}".format(topology_name))
        n_qubits = int(ma.group(1))
        return cls(n_qubits=n_qubits)


SYC23_GRAPH = nx.from_edgelist([
    ((3, 2), (4, 2)), ((4, 1), (5, 1)), ((4, 2), (4, 1)),
    ((4, 2), (4, 3)), ((4, 2), (5, 2)), ((4, 3), (5, 3)),
    ((5, 1), (5, 0)), ((5, 1), (5, 2)), ((5, 1), (6, 1)),
    ((5, 2), (5, 3)), ((5, 2), (6, 2)), ((5, 3), (5, 4)),
    ((5, 3), (6, 3)), ((5, 4), (6, 4)), ((6, 1), (6, 2)),
    ((6, 2), (6, 3)), ((6, 2), (7, 2)), ((6, 3), (6, 4)),
    ((6, 3), (7, 3)), ((6, 4), (6, 5)), ((6, 4), (7, 4)),
    ((6, 5), (7, 5)), ((7, 2), (7, 3)), ((7, 3), (7, 4)),
    ((7, 3), (8, 3)), ((7, 4), (7, 5)), ((7, 4), (8, 4)),
    ((7, 5), (7, 6)), ((7, 5), (8, 5)), ((8, 3), (8, 4)),
    ((8, 4), (8, 5)), ((8, 4), (9, 4)),
])


class DeviceBasedTopology:
    def __init__(self, graph, assoc_processor_id, epoch=1):
        self.graph = graph
        self.n_qubits = graph.number_of_nodes()
        self.assoc_processor_id = assoc_processor_id
        self.epoch = epoch

        if epoch != 1:
            epoch_str = f'-mk{epoch}'
        else:
            epoch_str = ''

        self.name = f'{assoc_processor_id}{epoch_str}'


RAINBOW_TOPOLOGY = DeviceBasedTopology(
    graph=SYC23_GRAPH,
    assoc_processor_id='rainbow',
    epoch=1
)

SYC50_GRAPH = nx.from_edgelist([
    ((0, 6), (0, 5)), ((1, 4), (1, 5)), ((1, 5), (0, 5)), ((1, 6), (0, 6)),
    ((1, 6), (1, 5)), ((1, 7), (1, 6)), ((2, 3), (2, 4)), ((2, 3), (3, 3)),
    ((2, 4), (1, 4)), ((2, 5), (1, 5)), ((2, 5), (2, 4)), ((2, 5), (2, 6)),
    ((2, 6), (1, 6)), ((3, 3), (3, 2)), ((3, 3), (4, 3)), ((3, 4), (2, 4)),
    ((3, 4), (3, 3)), ((3, 4), (4, 4)), ((3, 5), (2, 5)), ((3, 5), (3, 4)),
    ((3, 5), (3, 6)), ((3, 5), (4, 5)), ((3, 6), (2, 6)), ((3, 6), (3, 7)),
    ((3, 6), (4, 6)), ((3, 7), (3, 8)), ((3, 8), (2, 8)), ((3, 9), (3, 8)),
    ((4, 1), (5, 1)), ((4, 2), (3, 2)), ((4, 2), (4, 1)), ((4, 3), (4, 2)),
    ((4, 3), (4, 4)), ((4, 5), (4, 4)), ((4, 5), (4, 6)), ((4, 5), (5, 5)),
    ((4, 6), (4, 7)), ((4, 6), (5, 6)), ((4, 7), (3, 7)), ((4, 9), (3, 9)),
    ((5, 1), (5, 0)), ((5, 2), (4, 2)), ((5, 2), (5, 1)), ((5, 3), (4, 3)),
    ((5, 3), (5, 2)), ((5, 3), (5, 4)), ((5, 3), (6, 3)), ((5, 4), (4, 4)),
    ((5, 5), (5, 4)), ((5, 6), (5, 5)), ((5, 6), (6, 6)), ((6, 1), (5, 1)),
    ((6, 2), (5, 2)), ((6, 2), (6, 1)), ((6, 2), (6, 3)), ((6, 4), (5, 4)),
    ((6, 4), (6, 3)), ((6, 4), (7, 4)), ((6, 5), (5, 5)), ((6, 5), (6, 4)),
    ((6, 5), (6, 6)), ((6, 5), (7, 5)), ((6, 6), (6, 7)), ((6, 6), (7, 6)),
    ((7, 2), (6, 2)), ((7, 2), (7, 3)), ((7, 3), (6, 3)), ((7, 3), (7, 4)),
    ((7, 3), (8, 3)), ((7, 5), (7, 4)), ((7, 5), (7, 6)), ((7, 5), (8, 5)),
    ((8, 4), (7, 4)), ((8, 4), (8, 3)), ((8, 4), (8, 5)), ((8, 4), (9, 4)),
])

RAINBOW_MK2_TOPOLOGY = DeviceBasedTopology(
    graph=SYC50_GRAPH,
    assoc_processor_id='rainbow',
    epoch=2
)

DEVICE_BASED_TOPOLOGIES = {
    'rainbow': RAINBOW_TOPOLOGY,
    'rainbow-mk2': RAINBOW_MK2_TOPOLOGY,
}


class GridTopology:
    def __init__(self, width, height):
        self.graph = nx.grid_2d_graph(height, width)
        self.n_qubits = width * height
        self.name = f'{width}x{height}-grid'

    @classmethod
    def from_name(cls, topology_name):
        ma = re.match(r'(\d+)x(\d+)-grid', topology_name)
        if ma is None:
            raise ValueError("Could not parse grid topology name: {}".format(topology_name))
        width = int(ma.group(1))
        height = int(ma.group(2))
        return cls(width=width, height=height)


def get_named_topology(topology_name: str):
    if topology_name in DEVICE_BASED_TOPOLOGIES:
        return DEVICE_BASED_TOPOLOGIES[topology_name]

    if topology_name.endswith('line'):
        return LineTopology.from_name(topology_name)

    if topology_name.endswith('grid'):
        return GridTopology.from_name(topology_name)


def get_first_subgraph_monomorphism(big_graph, small_graph):
    matcher = GraphMatcher(big_graph, small_graph)
    mapping_big_to_small = next(matcher.subgraph_monomorphisms_iter())
    mapping_small_to_big = {v: k for k, v in mapping_big_to_small.items()}
    placed_graph = nx.relabel_nodes(small_graph, mapping_small_to_big)


def get_all_subgraph_monomorphisms(big_graph, small_graph):
    matcher = GraphMatcher(big_graph, small_graph)
    mappings_big_to_small = list(matcher.subgraph_monomorphisms_iter())
    return mappings_big_to_small


def count_subgraph_monomorphisms(big_graph, small_graph):
    matcher = GraphMatcher(big_graph, small_graph)
    return sum(1 for _ in matcher.subgraph_monomorphisms_iter())

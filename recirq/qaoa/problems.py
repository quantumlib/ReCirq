# Copyright 2020 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List, Union

import networkx as nx
import numpy as np

import cirq
import recirq

import scipy.stats


def _validate_problem_graph(graph: nx.Graph):
    """Validation logic for `GraphProblem`s.

    Checks that there are no orphan nodes and all edges have a "weight"
    attribute.
    """
    # TODO: while fixing nodes, this may happen as an intermediate.
    # if min(graph.degree[node] for node in graph.nodes) == 0:
    #     raise ValueError("Can't serialize graphs with orphan nodes")

    all_attrs = set()
    for u, v in graph.edges:
        all_attrs |= graph.edges[u, v].keys()
    if graph.number_of_edges() > 0 and not all_attrs == {'weight'}:
        raise ValueError(
            "Problem graph must have `weight` edge attributes")

    if sorted(graph.nodes) != list(range(graph.number_of_nodes())):
        raise ValueError(
            "Problem graph must have contiguous, 0-indexed integer nodes, not {}".format(
                graph.nodes))


def _graph_to_serializable_edgelist(graph: nx.Graph):
    """Helper function for making XXProblem instances JSON serializable."""
    _validate_problem_graph(graph)
    return [(v1, v2, float(w)) for (v1, v2, w) in graph.edges.data('weight')]


def _serialized_edgelist_to_graph(edgelist: List[Tuple]):
    """Helper function for deserializing XXProblem instances from JSON."""
    g = nx.Graph()
    for u, v, w in edgelist:
        g.add_edge(u, v, weight=w)
    return g


def get_growing_subgraphs(device_graph: nx.Graph, central_qubit: cirq.Qid,
                          min_size=2, max_size=None) -> Dict[int, Tuple[cirq.Qid]]:
    """Start with a central qubit and grow a subgraph radially.

    Qubits are added by path length to the central qubit. For qubits
    with equivalent path lengths, they are added according to the qubits'
    sorted order.

    Args:
        device_graph: The graph with qubits as nodes
        central_qubit: Where to start growing the subgraphs from
        min_size: The minimum size of resulting subgraphs
        max_size: The maximum size of resulting subgraphs, default: no max

    Returns:
        A dictionary from number of qubits to the subgraph expressed as a
        tuple of Qids.
    """
    by_radius = defaultdict(list)
    for q, distance in nx.shortest_path_length(device_graph, source=central_qubit).items():
        by_radius[distance].append(q)
    by_radius = {k: sorted(v) for k, v in by_radius.items()}

    subgraphs = []
    for r in by_radius:
        qs = by_radius[r]
        for i, q in enumerate(qs):
            if len(subgraphs) > 0:
                prev = subgraphs[-1]
            else:
                prev = tuple()

            subgraph = prev + (q,)
            subgraphs.append(subgraph)

    if max_size is None:
        max_size = device_graph.number_of_nodes()

    return {len(subgraph): subgraph for subgraph in subgraphs
            if min_size <= len(subgraph) <= max_size}


def random_plus_minus_1_weights(graph: nx.Graph, rs: Optional[np.random.RandomState] = None):
    """Take a graph and make an equivalent graph with weights of plus or minus
    one on each edge.

    Args:
        graph: A graph to add weights to
        rs: A RandomState for making replicable experiments. If not provided,
            the global numpy random state will be used. Please be careful
            when generating random problems. You should construct *one*
            seeded RandomState at the beginning of your script and use
            that one RandomState (in a deterministic fashion) to generate
            all the problem instances you may need.
    """
    if rs is None:
        rs = np.random
    elif not isinstance(rs, np.random.RandomState):
        raise ValueError("Invalid random state: {}".format(rs))

    problem_graph = nx.Graph()
    for n1, n2 in graph.edges:
        problem_graph.add_edge(n1, n2, weight=rs.choice([-1, 1]))
    return problem_graph


@dataclass
class HardwareGridProblem:
    graph: nx.Graph
    coordinates: List[Tuple[int, int]]

    def _json_dict_(self):
        return {
            'cirq_type': 'recirq.qaoa.' + self.__class__.__name__,
            'edgelist': _graph_to_serializable_edgelist(self.graph),
            'coordinates': self.coordinates,
        }

    @classmethod
    def _from_json_dict_(cls, edgelist, coordinates, **kwargs):
        graph = _serialized_edgelist_to_graph(edgelist)
        coordinates = [tuple(v) for v in coordinates]
        return cls(graph=graph, coordinates=coordinates)


recirq.Registry.register('recirq.qaoa.HardwareGridProblem', HardwareGridProblem)


def get_all_hardware_grid_problems(
        device_graph: nx.Graph,
        central_qubit: cirq.GridQubit,
        n_instances: int,
        rs: np.random.RandomState,
):
    """Helper function to get all subgraphs for a given named device.

    This is annotated with lru_cache so you can freely call this function
    multiple times without re-constructing the list of qubits.

    Used by `generate_hardware_problem_problem` to get a subgraph for a given
    value of n_qubits.

    Returns:
        A dictionary indexed by n_qubit, instance_i
    """
    all_hg_problems: Dict[Tuple[int, int], HardwareGridProblem] = {}
    subgraphs = get_growing_subgraphs(device_graph=device_graph,
                                      central_qubit=central_qubit)
    for n_qubits in sorted(subgraphs):
        subgraph = nx.subgraph(device_graph, subgraphs[n_qubits])
        for instance_i in range(n_instances):
            problem = random_plus_minus_1_weights(subgraph, rs=rs)
            qubits = sorted(problem.nodes)
            coordinates = [(q.row, q.col) for q in qubits]
            problem = nx.relabel_nodes(problem, {q: i for i, q in enumerate(qubits)})

            all_hg_problems[n_qubits, instance_i] = HardwareGridProblem(
                graph=problem,
                coordinates=coordinates,
            )

    return all_hg_problems


@dataclass
class SKProblem:
    graph: nx.Graph

    def _json_dict_(self):
        return {
            'cirq_type': 'recirq.qaoa.' + self.__class__.__name__,
            'edgelist': _graph_to_serializable_edgelist(self.graph)
        }

    @classmethod
    def _from_json_dict_(cls, edgelist, **kwargs):
        return cls(graph=_serialized_edgelist_to_graph(edgelist))


recirq.Registry.register('recirq.qaoa.SKProblem', SKProblem)


def get_all_sk_problems(
        max_n_qubits: int,
        n_instances: int,
        rs: np.random.RandomState):
    """Helper function to get all random Sherrington-Kirkpatrick problem
    instances for a given number of qubits.

    This is annotated with lru_cache so you can freely call this function
    multiple times without re-sampling random weights. This function uses
    a seeded `np.random.RandomState`, so output is deterministic.
    """
    all_sk_problems: Dict[Tuple[int, int], SKProblem] = {}
    for n_qubits in range(2, max_n_qubits + 1):
        for instance_i in range(n_instances):
            graph = nx.complete_graph(n_qubits)
            graph = random_plus_minus_1_weights(graph, rs=rs)

            all_sk_problems[n_qubits, instance_i] = SKProblem(
                graph=graph,
            )
    return all_sk_problems


@dataclass
class ThreeRegularProblem:
    graph: nx.Graph

    def _json_dict_(self):
        return {
            'cirq_type': 'recirq.qaoa.' + self.__class__.__name__,
            'edgelist': _graph_to_serializable_edgelist(self.graph)
        }

    @classmethod
    def _from_json_dict_(cls, edgelist, **kwargs):
        return cls(graph=_serialized_edgelist_to_graph(edgelist))


recirq.Registry.register('recirq.qaoa.ThreeRegularProblem', ThreeRegularProblem)


def get_all_3_regular_problems(
        max_n_qubits: int,
        n_instances: int,
        rs: np.random.RandomState):
    """Helper function to get all random 3-regular graph problem
    instances for a given number of qubits.

    This is annotated with lru_cache so you can freely call this function
    multiple times without re-sampling random weights. This function uses
    a seeded `np.random.RandomState`, so output is deterministic.
    """
    all_3reg_problems: Dict[Tuple[int, int], ThreeRegularProblem] = {}
    for n_qubits in range(4, max_n_qubits + 1, 2):
        for instance_i in range(n_instances):
            graph = nx.random_regular_graph(3, n_qubits, seed=rs)
            nx.set_edge_attributes(graph, values=1, name='weight')
            all_3reg_problems[n_qubits, instance_i] = ThreeRegularProblem(
                graph=graph)
    return all_3reg_problems


ProblemT = Union[HardwareGridProblem, SKProblem, ThreeRegularProblem]


def asymmetric_coupling_ferromagnet_chain(n: int, rs: Union[np.random.RandomState],
                                          other_coupling: float = 0.5, *, shuffle=True):
    g = nx.Graph()
    n_edges = n - 1
    couplings = [-1] * (n_edges // 2) + [-other_coupling] * (n_edges - n_edges // 2)
    if shuffle:
        rs.shuffle(couplings)
    for i1, i2, J in zip(range(n), range(1, n), couplings):
        g.add_edge(i1, i2, weight=J)
    return g


def asymmetric_coupling_ferromagnet_grid(n: int, rs: Union[np.random.RandomState],
                                         other_coupling: float = 0.5, *, shuffle=True):
    width = int(np.ceil(np.sqrt(n)).item())
    height = n // width
    assert width * height == n, (width, height)

    g = nx.grid_2d_graph(height, width)
    n_edges = g.number_of_edges()
    couplings = [-1] * (n_edges // 2) + [-other_coupling] * (n_edges - n_edges // 2)
    if shuffle:
        rs.shuffle(couplings)

    def colindex(edge):
        n1, n2 = edge
        r1, c1 = n1
        r2, c2 = n2
        return (c1, c2)

    nx.set_edge_attributes(g, {
        e: coupling
        for e, coupling in zip(sorted(g.edges, key=colindex), couplings)
    }, name='weight')
    g = nx.convert_node_labels_to_integers(g)
    return g


def asymmetric_coupling_3reg(n: int, rs: np.random.RandomState,
                             other_coupling: float = 0.5):
    graph = nx.random_regular_graph(d=3, n=n, seed=rs)
    n_edges = graph.number_of_edges()
    couplings = [1] * (n_edges // 2) + [other_coupling] * (n_edges - n_edges // 2)
    rs.shuffle(couplings)
    nx.set_edge_attributes(graph, values={
        (i1, i2): c
        for (i1, i2), c in zip(graph.edges, couplings)
    }, name='weight')
    return graph


def beta_distributed_sk(n: int, rs: np.random.RandomState, shape_param):
    graph = nx.complete_graph(n)
    bdist = scipy.stats.beta(a=shape_param, b=shape_param, loc=-1, scale=2)
    couplings = bdist.rvs(graph.number_of_edges(), random_state=rs)

    nx.set_edge_attributes(graph, values={
        (i1, i2): c
        for (i1, i2), c in zip(graph.edges, couplings)
    }, name='weight')
    return graph


def gaussian_sk(n: int, rs: np.random.RandomState):
    graph = nx.complete_graph(n)
    n_edges = graph.number_of_edges()

    couplings = rs.normal(size=n_edges)
    nx.set_edge_attributes(graph, values={
        (i1, i2): c
        for (i1, i2), c in zip(sorted(graph.edges), couplings)
    }, name='weight')
    return graph


def partially_rounded_sk(n: int, rs: np.random.RandomState,
                         round_factor: float = 0.5):
    graph = nx.complete_graph(n)
    n_edges = graph.number_of_edges()

    normal_couplings = rs.normal(size=n_edges)
    coupling_signs = np.sign(normal_couplings)
    couplings = round_factor * coupling_signs + (1 - round_factor) * normal_couplings
    nx.set_edge_attributes(graph, values={
        (i1, i2): c
        for (i1, i2), c in zip(sorted(graph.edges), couplings)
    }, name='weight')
    return graph

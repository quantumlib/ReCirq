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
import networkx as nx
import cirq
from recirq.qaoa.problems import random_plus_minus_1_weights, _validate_problem_graph, \
    _graph_to_serializable_edgelist, _serialized_edgelist_to_graph, get_growing_subgraphs, \
    get_all_hardware_grid_problems, HardwareGridProblem, get_all_sk_problems, SKProblem, \
    get_all_3_regular_problems, ThreeRegularProblem
import pytest
import numpy as np


def test_validate_problem_qubit_nodes():
    def random_sk_model_with_qubit_nodes(n: int):
        graph = nx.complete_graph(n)
        graph = nx.relabel_nodes(
            graph,
            mapping={i: cirq.LineQubit(i) for i in range(n)})
        return random_plus_minus_1_weights(graph)

    problem_graph = random_sk_model_with_qubit_nodes(n=3)
    with pytest.raises(ValueError) as e:
        _validate_problem_graph(problem_graph)
    assert e.match(r'Problem graph must have contiguous, 0-indexed integer nodes.*')


def test_validate_problem_no_weight():
    problem = nx.complete_graph(n=3)
    with pytest.raises(ValueError) as e:
        _validate_problem_graph(problem)
    assert e.match(r'Problem graph must have `weight` edge attributes.*')


def test_graph_to_serializable_edgelist():
    g = random_plus_minus_1_weights(nx.complete_graph(4), rs=np.random.RandomState(52))
    edgelist = _graph_to_serializable_edgelist(g)
    assert isinstance(edgelist, list)
    assert isinstance(edgelist[0], tuple)
    assert len(edgelist) == 4 * 3 / 2
    for n1, n2, w in edgelist:
        assert 0 <= n1 < 4
        assert 0 <= n2 < 4
        assert w in [-1, 1]


def test_serialized_edgelist_to_graph():
    edgelist = [
        (0, 1, 1.0),
        (0, 2, -1.0),
        (0, 3, 1.0),
        (1, 2, 1.0),
        (1, 3, 1.0),
        (2, 3, -1.0)
    ]
    g = random_plus_minus_1_weights(nx.complete_graph(4), rs=np.random.RandomState(52))
    assert nx.is_isomorphic(_serialized_edgelist_to_graph(edgelist), g)


def test_get_growing_subgraphs():
    device_graph = nx.grid_2d_graph(3, 3)
    device_graph = nx.relabel_nodes(
        device_graph, mapping={(r, c): cirq.GridQubit(r, c)
                               for r, c in device_graph.nodes})
    growing_subgraphs = get_growing_subgraphs(device_graph, central_qubit=cirq.GridQubit(1, 1))
    assert set(growing_subgraphs.keys()) == set(range(2, 9 + 1))
    for k, v in growing_subgraphs.items():
        assert len(v) == k


def test_random_plus_minus_1_weights():
    g1 = nx.complete_graph(4)
    g2 = random_plus_minus_1_weights(g1, rs=np.random.RandomState(53))
    assert g1 != g2  # makes a new graph
    assert nx.is_isomorphic(g1, g2, node_match=lambda x, y: x == y)
    assert not nx.is_isomorphic(g1, g2, edge_match=lambda x, y: x == y)

    for u, v, w in g2.edges.data('weight'):
        assert 0 <= u <= 4
        assert 0 <= v <= 4
        assert w in [-1, 1]


def test_get_all_hardware_grid_problems():
    device_graph = nx.grid_2d_graph(3, 3)
    device_graph = nx.relabel_nodes(
        device_graph, mapping={(r, c): cirq.GridQubit(r, c)
                               for r, c in device_graph.nodes})
    problems = get_all_hardware_grid_problems(
        device_graph, central_qubit=cirq.GridQubit(1, 1), n_instances=3,
        rs=np.random.RandomState(52))
    keys_should_be = [(n, i) for n in range(2, 9 + 1) for i in range(3)]
    assert list(problems.keys()) == keys_should_be
    for (n, i), v in problems.items():
        assert isinstance(v, HardwareGridProblem)
        assert len(v.graph.nodes) == n
        assert len(v.coordinates) == n
        for r, c in v.coordinates:
            assert 0 <= r < 3
            assert 0 <= c < 3


def test_get_all_sk_problems():
    problems = get_all_sk_problems(
        max_n_qubits=5, n_instances=3, rs=np.random.RandomState())
    keys_should_be = [(n, i) for n in range(2, 5 + 1) for i in range(3)]
    assert list(problems.keys()) == keys_should_be
    for (n, i), problem in problems.items():
        assert isinstance(problem, SKProblem)
        assert len(problem.graph.nodes) == n
        assert problem.graph.number_of_edges() == (n * (n - 1)) / 2


def test_get_all_3_regular_problems():
    problems = get_all_3_regular_problems(
        max_n_qubits=8, n_instances=3, rs=np.random.RandomState())
    keys_should_be = [(n, i) for n in range(4, 8 + 1)
                      for i in range(3) if n * 3 % 2 == 0]
    assert list(problems.keys()) == keys_should_be
    for (n, i), problem in problems.items():
        assert isinstance(problem, ThreeRegularProblem)
        assert set(degree for node, degree in problem.graph.degree) == {3}

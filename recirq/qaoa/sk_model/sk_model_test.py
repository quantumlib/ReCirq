# Copyright 2022 Google
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
import numpy as np
import pytest

import cirq
from recirq.qaoa.sk_model.sk_model import _graph_from_row_major_upper_triangular, \
    _all_to_all_couplings_from_graph, SKModelQAOASpec, sk_model_qaoa_spec_to_exe


def test_graph_from_row_major_upper_triangular():
    couplings = np.array([
        [0, 1, 2, 3],
        [0, 0, 4, 5],
        [0, 0, 0, 6],
        [0, 0, 0, 0],
    ])
    flat_couplings = np.arange(1, 6 + 1)

    graph1 = nx.from_numpy_array(couplings)
    graph2 = _graph_from_row_major_upper_triangular(flat_couplings, n=4)
    assert sorted(graph1.nodes) == sorted(graph2.nodes)
    assert sorted(graph1.edges) == sorted(graph2.edges)

    for u, v, w in graph1.edges.data('weight'):
        assert w == graph2.edges[u, v]['weight']


def test_graph_from_row_major_upper_triangular_bad():
    with pytest.raises(ValueError):
        _graph_from_row_major_upper_triangular([1, 2, 3, 4], n=2)


def test_all_to_all_couplings_from_graph():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    g.add_edge(0, 2, weight=2)
    g.add_edge(1, 2, weight=3)
    couplings = _all_to_all_couplings_from_graph(g)
    assert couplings == (1, 2, 3)


def test_all_to_all_couplings_from_graph_missing():
    g = nx.Graph()
    g.add_edge(0, 1, weight=1)
    # g.add_edge(0, 2, weight=2)
    g.add_edge(1, 2, weight=3)
    with pytest.raises(KeyError):
        _ = _all_to_all_couplings_from_graph(g)


def test_all_to_all_couplings_from_graph_bad_nodes():
    g = nx.Graph()
    g.add_edge(10, 11, weight=1)
    g.add_edge(10, 12, weight=2)
    g.add_edge(11, 12, weight=3)
    with pytest.raises(ValueError):
        _ = _all_to_all_couplings_from_graph(g)


@pytest.mark.parametrize('n', [5, 6])
def test_graph_round_trip(n):
    couplings = tuple(np.random.choice([0, 1], size=n * (n - 1) // 2))
    assert couplings == _all_to_all_couplings_from_graph(
        _graph_from_row_major_upper_triangular(couplings, n=n))


def test_spec_to_exe():
    spec = SKModelQAOASpec(
        n_nodes=3, all_to_all_couplings=[1, -1, 1], p_depth=1, n_repetitions=1_000
    )
    assert isinstance(spec.all_to_all_couplings, tuple)
    assert hash(spec) is not None
    exe = sk_model_qaoa_spec_to_exe(spec)
    init_hadamard_depth = 1
    zz_swap_depth = 2  # zz + swap
    driver_depth = 1
    measure_depth = 1
    assert len(
        exe.circuit) == init_hadamard_depth + zz_swap_depth * 3 + driver_depth + measure_depth
    assert exe.spec == spec
    assert exe.problem_topology == cirq.LineTopology(3)

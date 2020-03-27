from typing import Dict, Sequence, cast, List

import numpy as np
import pytest

import cirq
import cirq.contrib.acquaintance as cca
import cirq.contrib.routing as ccr
import networkx as nx

from cirq.google import Sycamore23
from recirq.qaoa.placement import place_line_on_device


def permute_gate(qubits: Sequence[cirq.Qid], permutation: List[int]):
    return cca.LinearPermutationGate(
        num_qubits=len(qubits),
        permutation={i: permutation[i] for i in range(len(permutation))}
    ).on(*qubits)


def test_place_on_grid():
    graph = nx.random_regular_graph(d=3, n=10)
    nx.set_edge_attributes(graph, values=1, name='weight')
    graph = random_3_regular_model(n=10)
    circuit_qubits = cirq.LineQubit.range(10)
    gammas = np.random.randn(2)
    betas = np.random.randn(2)
    circuit = circuits.qaoa_circuit(graph, gammas, betas)
    routed_circuit, initial_qubit_map, final_qubit_map = (
        placement.place_on_grid(circuit, n_rows=4, n_cols=3))

    # Check that constraints are not violated
    for _, op, _ in routed_circuit.findall_operations_with_gate_type(cirq.TwoQubitGate):
        a, b = op.qubits
        a = cast(cirq.GridQubit, a)
        b = cast(cirq.GridQubit, b)
        assert a.is_adjacent(b)

    # Check that the circuits are equivalent
    final_to_initial_qubit_map = {final_qubit_map[cq]: initial_qubit_map[cq]
                                  for cq in circuit_qubits}
    initial_qubits = [initial_qubit_map[cq] for cq in circuit_qubits]
    final_permutation = [initial_qubits.index(final_to_initial_qubit_map[q])
                         for q in initial_qubits]
    rcircuit_with_perm = routed_circuit.copy()
    rcircuit_with_perm.append(permute_gate(initial_qubits, final_permutation))
    expected = circuit.unitary(qubit_order=cirq.QubitOrder.explicit(circuit_qubits))
    actual = rcircuit_with_perm.unitary(qubit_order=cirq.QubitOrder.explicit(initial_qubits))
    cirq.testing.assert_allclose_up_to_global_phase(expected, actual, atol=1e-8)


def test_place_on_device():
    graph = graph_models.random_3_regular_model(n=10)
    circuit_qubits = cirq.LineQubit.range(10)
    gammas = np.random.randn(2)
    betas = np.random.randn(2)
    circuit = circuits.qaoa_circuit(graph, gammas, betas)
    device = cirq.google.Bristlecone
    routed_circuit, initial_qubit_map, final_qubit_map = (
        placement.place_on_device(circuit, device))

    # Check that constraints are not violated
    for _, op, _ in routed_circuit.findall_operations_with_gate_type(cirq.TwoQubitGate):
        a, b = op.qubits
        a = cast(cirq.GridQubit, a)
        b = cast(cirq.GridQubit, b)
        assert a.is_adjacent(b)

    # Check that the circuits are equivalent
    final_to_initial_qubit_map = {final_qubit_map[cq]: initial_qubit_map[cq]
                                  for cq in circuit_qubits}
    initial_qubits = [initial_qubit_map[cq] for cq in circuit_qubits]
    final_permutation = [initial_qubits.index(final_to_initial_qubit_map[q])
                         for q in initial_qubits]
    rcircuit_with_perm = routed_circuit.copy()
    rcircuit_with_perm.append(permute_gate(initial_qubits, final_permutation))
    expected = circuit.unitary(qubit_order=cirq.QubitOrder.explicit(circuit_qubits))
    actual = rcircuit_with_perm.unitary(qubit_order=cirq.QubitOrder.explicit(initial_qubits))
    cirq.testing.assert_allclose_up_to_global_phase(expected, actual, atol=1e-8)


def test_min_weight_simple_paths_brute_force():
    test_graph = nx.grid_2d_graph(4, 4)
    test_graph.remove_node((3, 0))
    test_graph.remove_node((0, 3))
    for e in test_graph.edges:
        test_graph[e[0]][e[1]]['weight'] = np.random.rand()

    bp_brute = placement.min_weight_simple_paths_brute_force(test_graph)
    for n in range(2, 14):
        assert nx.is_simple_path(test_graph, bp_brute[n])

    assert 14 not in bp_brute


def test_min_weight_simple_path_greedy():
    test_graph = nx.grid_2d_graph(4, 4)
    test_graph.remove_node((3, 0))
    test_graph.remove_node((0, 3))
    for e in test_graph.edges:
        test_graph[e[0]][e[1]]['weight'] = np.random.rand()

    weights = [w for u, v, w in test_graph.edges.data('weight')]

    # it better return the lowest weight edge for a path consisting of 2 nodes
    path = placement.min_weight_simple_path_greedy(test_graph, 2)
    assert placement.path_weight(test_graph, path) == min(weights)

    # it should return simple paths
    path = placement.min_weight_simple_path_greedy(test_graph, 5)
    assert nx.is_simple_path(test_graph, path)

    # there should not exist a simple path of 14 nodes
    assert placement.min_weight_simple_path_greedy(test_graph, 14) is None


def test_min_weight_simple_path_anneal():
    test_graph = nx.grid_2d_graph(4, 4)
    test_graph.remove_node((3, 0))
    test_graph.remove_node((0, 3))
    for e in test_graph.edges:
        test_graph[e[0]][e[1]]['weight'] = np.random.rand()

    # it should return simple paths
    path = placement.min_weight_simple_path_anneal(test_graph, 5)
    assert nx.is_simple_path(test_graph, path)

    # there should not exist a simple path of 14 nodes
    assert placement.min_weight_simple_path_anneal(test_graph, 14) is None


def _fake_calib_data():
    err_graph = ccr.xmon_device_to_graph(Sycamore23)
    nx.set_edge_attributes(err_graph, 0.005, 'weight')
    nx.set_node_attributes(err_graph, 0.05, 'weight')
    return err_graph


@pytest.mark.parametrize('n', [3, 8, 13])
@pytest.mark.parametrize('method', ['brute_force', 'random', 'greedy',
                                    'anneal', 'mst', 'mixed'])
def test_on_device(n, method):
    err_graph = _fake_calib_data()
    path = place_line_on_device('Sycamore23', n=n,
                                line_placement_strategy=method,
                                err_graph=err_graph)
    if n == 13:
        if method == 'greedy' or method == 'mst':
            assert path is None
            return

    assert nx.is_simple_path(err_graph, path)

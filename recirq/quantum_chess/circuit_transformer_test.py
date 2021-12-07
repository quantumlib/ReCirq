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
import pytest
import cirq
import cirq_google as cg

import recirq.quantum_chess.circuit_transformer as ct
import recirq.quantum_chess.quantum_moves as qm

a1 = cirq.NamedQubit("a1")
a2 = cirq.NamedQubit("a2")
a3 = cirq.NamedQubit("a3")
a4 = cirq.NamedQubit("a4")
b1 = cirq.NamedQubit("b1")
b2 = cirq.NamedQubit("b2")
b3 = cirq.NamedQubit("b3")
c1 = cirq.NamedQubit("c1")
c2 = cirq.NamedQubit("c2")
c3 = cirq.NamedQubit("c3")
d1 = cirq.NamedQubit("d1")


@pytest.mark.parametrize("device", (cg.Sycamore23, cg.Sycamore))
def test_qubits_within(device):
    """Coupling graph of grid qubits looks like:

    Q0 = Q1 = Q2
    ||   ||   ||
    Q3 = Q4 = Q5
    """
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    grid_qubits = cirq.GridQubit.rect(2, 3)
    assert transformer.qubits_within(0, grid_qubits[0], grid_qubits, set()) == 1
    assert transformer.qubits_within(1, grid_qubits[0], grid_qubits, set()) == 3
    assert transformer.qubits_within(2, grid_qubits[0], grid_qubits, set()) == 5
    assert transformer.qubits_within(3, grid_qubits[0], grid_qubits, set()) == 6
    assert transformer.qubits_within(10, grid_qubits[0], grid_qubits, set()) == 6


@pytest.mark.parametrize("device", (cg.Sycamore23, cg.Sycamore))
def test_edges_within(device):
    """
    The circuit looks like:

    a1 --- a4 --- a3 --- a2      d1
           |      |      |
           b2     b1 --- c3
    """
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    circuit = cirq.Circuit(
        cirq.X(d1),
        cirq.ISWAP(a1, a4) ** 0.5,
        cirq.ISWAP(a4, b2),
        cirq.ISWAP(a4, a3),
        cirq.ISWAP(a3, b1),
        cirq.ISWAP(a3, a2),
        cirq.ISWAP(b1, c3),
        cirq.ISWAP(a2, c3),
    )
    graph = {}
    for moment in circuit:
        for op in moment:
            if len(op.qubits) == 2:
                q1, q2 = op.qubits
                if q1 not in graph:
                    graph[q1] = []
                if q2 not in graph:
                    graph[q2] = []
                if q2 not in graph[q1]:
                    graph[q1].append(q2)
                if q1 not in graph[q2]:
                    graph[q2].append(q1)
    assert transformer.edges_within(0, cirq.NamedQubit("a3"), graph, set()) == 1
    assert transformer.edges_within(1, cirq.NamedQubit("a3"), graph, set()) == 4
    assert transformer.edges_within(2, cirq.NamedQubit("a3"), graph, set()) == 7
    assert transformer.edges_within(10, cirq.NamedQubit("a3"), graph, set()) == 7


@pytest.mark.parametrize(
    "transformer",
    [
        ct.ConnectivityHeuristicCircuitTransformer,
        ct.DynamicLookAheadHeuristicCircuitTransformer,
    ],
)
@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_single_qubit_ops(transformer, device):
    c = cirq.Circuit(cirq.X(a1), cirq.X(a2), cirq.X(a3))
    t = transformer(device)
    device.validate_circuit(t.transform(c))


@pytest.mark.parametrize(
    "transformer",
    [
        ct.ConnectivityHeuristicCircuitTransformer,
        ct.DynamicLookAheadHeuristicCircuitTransformer,
    ],
)
@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_single_qubit_and_two_qubits_ops(transformer, device):
    c = cirq.Circuit(cirq.X(a1), cirq.X(a2), cirq.X(a3), cirq.ISWAP(a3, a4) ** 0.5)
    t = transformer(device)
    device.validate_circuit(t.transform(c))


@pytest.mark.parametrize(
    "transformer",
    [
        ct.ConnectivityHeuristicCircuitTransformer,
        ct.DynamicLookAheadHeuristicCircuitTransformer,
    ],
)
@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_three_split_moves(transformer, device):
    c = cirq.Circuit(
        qm.split_move(a1, a2, b1), qm.split_move(a2, a3, b3), qm.split_move(b1, c1, c2)
    )
    t = transformer(device)
    device.validate_circuit(t.transform(c))


@pytest.mark.parametrize(
    "transformer",
    [
        ct.ConnectivityHeuristicCircuitTransformer,
        ct.DynamicLookAheadHeuristicCircuitTransformer,
    ],
)
@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_disconnected(transformer, device):
    c = cirq.Circuit(
        qm.split_move(a1, a2, a3),
        qm.split_move(a3, a4, d1),
        qm.split_move(b1, b2, b3),
        qm.split_move(c1, c2, c3),
    )
    t = transformer(device)
    device.validate_circuit(t.transform(c))


@pytest.mark.parametrize(
    "transformer",
    [
        ct.ConnectivityHeuristicCircuitTransformer,
        ct.DynamicLookAheadHeuristicCircuitTransformer,
    ],
)
@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_move_around_square(transformer, device):
    c = cirq.Circuit(
        qm.normal_move(a1, a2),
        qm.normal_move(a2, b2),
        qm.normal_move(b2, b1),
        qm.normal_move(b1, a1),
    )
    t = transformer(device)
    device.validate_circuit(t.transform(c))


@pytest.mark.parametrize(
    "transformer",
    [
        ct.ConnectivityHeuristicCircuitTransformer,
        ct.DynamicLookAheadHeuristicCircuitTransformer,
    ],
)
@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_split_then_merge(transformer, device):
    c = cirq.Circuit(
        qm.split_move(a1, a2, b1),
        qm.split_move(a2, a3, b3),
        qm.split_move(b1, c1, c2),
        qm.normal_move(c1, d1),
        qm.normal_move(a3, a4),
        qm.merge_move(a4, d1, a1),
    )
    t = transformer(device)
    device.validate_circuit(t.transform(c))


@pytest.mark.parametrize("device", [cg.Sycamore23, cg.Sycamore])
def test_split_then_merge_trapezoid(device):
    c = cirq.Circuit(
        qm.split_move(a1, a2, b1), qm.normal_move(a2, a3), qm.merge_move(a3, b1, b3)
    )
    t = ct.DynamicLookAheadHeuristicCircuitTransformer(device)
    device.validate_circuit(t.transform(c))


def test_too_many_qubits():
    c = cirq.Circuit()
    for i in range(24):
        c.append(cirq.X(cirq.NamedQubit("q" + str(i))))
    t = ct.ConnectivityHeuristicCircuitTransformer(cg.Sycamore23)
    with pytest.raises(ct.DeviceMappingError, match="Qubits exhausted"):
        t.transform(c)


def test_two_operations_on_single_qubit():
    """Tests that a new qubit is NOT allocated for every gate."""
    qubit = cirq.NamedQubit("a")
    c = cirq.Circuit(*[cirq.X(qubit)] * 99)
    device = cg.Sycamore23
    t = ct.ConnectivityHeuristicCircuitTransformer(device)
    device.validate_circuit(t.transform(c))


def test_sycamore_decomposer_reject_0_controlled():
    c = cirq.Circuit(cirq.X(a1).controlled_by(a2, control_values=[0]))
    decomposer = ct.SycamoreDecomposer()
    with pytest.raises(ct.DeviceMappingError):
        decomposer.optimize_circuit(c)

# Copyright 2025 Google
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

import cirq

from recirq.dfl.dfl_entropy import layer_floquet


def test_layer_floquet_basic():
    # Create a simple grid of 4 qubits
    grid = [cirq.GridQubit(0, i) for i in range(4)]
    dt = 0.1
    h = 0.5
    mu = 0.3

    circuit = layer_floquet(grid, dt, h, mu)
    # Check that the output is a cirq.Circuit
    assert isinstance(circuit, cirq.Circuit)
    # Check that the circuit has operations on all qubits
    qubits_in_circuit = set(q for op in circuit.all_operations() for q in op.qubits)
    assert set(grid) == qubits_in_circuit


def test_layer_floquet_three_qubits_structure():
    # Test for 3 qubits, manually construct the expected circuit and compare
    q1, q2, q3 = cirq.LineQubit.range(3)
    grid = [q1, q2, q3]
    dt = 0.2
    h = 0.4
    mu = 0.6

    # Manually construct the expected circuit based on layer_interaction and RX layer
    expected_circuit = cirq.Circuit(
        cirq.H(q2),
        cirq.CZ(q1, q2),
        cirq.CZ(q3, q2),
        cirq.rx(2 * dt).on(q2),
        cirq.CZ(q3, q2),
        cirq.CZ(q1, q2),
        cirq.H(q2),
    )

    expected_circuit += cirq.Circuit.from_moments(
        cirq.Moment(
            [
                cirq.rx(2 * mu * dt).on(q1),
                cirq.rx(2 * h * dt).on(q2),
                cirq.rx(2 * mu * dt).on(q3),
            ]
        )
    )

    actual_circuit = layer_floquet(grid, dt, h, mu)

    assert actual_circuit == expected_circuit

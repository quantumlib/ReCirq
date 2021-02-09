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

import recirq.quantum_chess.circuit_transformer as ct
import recirq.quantum_chess.quantum_moves as qm

a1 = cirq.NamedQubit('a1')
a2 = cirq.NamedQubit('a2')
a3 = cirq.NamedQubit('a3')
a4 = cirq.NamedQubit('a4')
b1 = cirq.NamedQubit('b1')
b2 = cirq.NamedQubit('b2')
b3 = cirq.NamedQubit('b3')
c1 = cirq.NamedQubit('c1')
c2 = cirq.NamedQubit('c2')
c3 = cirq.NamedQubit('c3')
d1 = cirq.NamedQubit('d1')


@pytest.mark.parametrize('device',
                         (cirq.google.Sycamore23, cirq.google.Sycamore))
def test_single_qubit_ops(device):
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    c = cirq.Circuit(cirq.X(a1), cirq.X(a2), cirq.X(a3))
    transformer.qubit_mapping(c)
    c = transformer.transform(c)
    device.validate_circuit(c)


@pytest.mark.parametrize('device',
                         (cirq.google.Sycamore23, cirq.google.Sycamore))
def test_single_qubit_with_two_qubits(device):
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    c = cirq.Circuit(cirq.X(a1), cirq.X(a2), cirq.X(a3),
                     cirq.ISWAP(a3, a4) ** 0.5)
    transformer.qubit_mapping(c)
    device.validate_circuit(transformer.transform(c))


@pytest.mark.parametrize('device',
                         (cirq.google.Sycamore23, cirq.google.Sycamore))
def test_three_split_moves(device):
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    c = cirq.Circuit(qm.split_move(a1, a2, b1), qm.split_move(a2, a3, b3),
                     qm.split_move(b1, c1, c2))
    transformer.qubit_mapping(c)
    device.validate_circuit(transformer.transform(c))


@pytest.mark.parametrize('device',
                         (cirq.google.Sycamore23, cirq.google.Sycamore))
def test_disconnected(device):
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    c = cirq.Circuit(qm.split_move(a1, a2, a3), qm.split_move(a3, a4, d1),
                     qm.split_move(b1, b2, b3), qm.split_move(c1, c2, c3))
    transformer.qubit_mapping(c)
    device.validate_circuit(transformer.transform(c))


@pytest.mark.parametrize('device',
                         (cirq.google.Sycamore23, cirq.google.Sycamore))
def test_move_around_square(device):
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    c = cirq.Circuit(qm.normal_move(a1, a2), qm.normal_move(a2, b2),
                     qm.normal_move(b2, b1), qm.normal_move(b1, a1))
    transformer.qubit_mapping(c)
    device.validate_circuit(transformer.transform(c))


@pytest.mark.parametrize('device',
                         (cirq.google.Sycamore23, cirq.google.Sycamore))
def test_split_then_merge(device):
    transformer = ct.ConnectivityHeuristicCircuitTransformer(device)
    c = cirq.Circuit(qm.split_move(a1, a2, b1), qm.split_move(a2, a3, b3),
                     qm.split_move(b1, c1, c2), qm.normal_move(c1, d1),
                     qm.normal_move(a3, a4), qm.merge_move(a4, d1, a1))
    transformer.qubit_mapping(c)
    device.validate_circuit(transformer.transform(c))

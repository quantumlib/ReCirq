# Copyright 2023 Google
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
import numpy as np
from openfermion import QubitOperator

from recirq.seniority_zero.misc import (
    add_echoes,
    check_if_sq_and_preserves_computational_basis,
    check_separated_single_and_two_qubit_layers,
    get_all_standard_qubit_sets,
    get_num_circuit_copies,
    get_operator_from_tag,
    get_standard_qubit_set,
    merge_faster,
    parallelize_circuits,
    qubit_hamming_distance,
    safe_concatenate_circuits,
    safe_tetris_circuits,
)


def make_test_circuit():
    qubits = [cirq.GridQubit(0, j) for j in range(5)]
    circuit = cirq.Circuit()
    for j in range(4):
        circuit.append(cirq.H(qubits[j + 1]))
        circuit.append(cirq.CZ(qubits[j], qubits[j + 1]))
        circuit.append(cirq.H(qubits[j + 1]))
    circuit.append(cirq.CZ(qubits[4], qubits[3]))
    for j in reversed(range(4)):
        circuit.append(cirq.H(qubits[j + 1]))
        circuit.append(cirq.CZ(qubits[j], qubits[j + 1]))
        circuit.append(cirq.H(qubits[j + 1]))
    return circuit


def test_get_operator_from_tag():
    tag = {'type': 'Z', 'ham_ids': [1]}
    op = get_operator_from_tag(tag)
    test_op = QubitOperator('Z1')
    assert op == test_op


def test_get_num_circuit_copies():
    tag = {'type': 'Z', 'ham_ids': [1]}
    hamiltonian = QubitOperator('Z1')
    factor = 1
    ncc = get_num_circuit_copies(tag, hamiltonian, factor)
    assert ncc == 1


def test_get_standard_qubit_set():
    qubits = get_standard_qubit_set(4)
    assert qubits[0] == cirq.GridQubit(0, 0)
    assert qubits[1] == cirq.GridQubit(0, 1)
    assert qubits[2] == cirq.GridQubit(1, 1)
    assert qubits[3] == cirq.GridQubit(1, 0)


def test_get_all_standard_qubit_sets():
    all_qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(3)]
    all_ladders = get_all_standard_qubit_sets(6, all_qubits)
    assert len(all_ladders) == 4


def test_qubit_hamming_distance():
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    assert qubit_hamming_distance(q0, q1) == 1


def test_check_if_sq_and_preserves_computational_basis():
    gate = cirq.rz(np.pi / 4).on(cirq.GridQubit(0, 0))
    assert check_if_sq_and_preserves_computational_basis(gate) is True
    gate = cirq.rx(np.pi / 4).on(cirq.GridQubit(0, 0))
    assert check_if_sq_and_preserves_computational_basis(gate) is False


def test_add_echoes():
    circuit = make_test_circuit()
    add_echoes(circuit)


def test_check_separated_single_and_two_qubit_layers():
    circuit = make_test_circuit()
    check_separated_single_and_two_qubit_layers([circuit])


def test_parallelize_circuits():
    c1 = cirq.Circuit([cirq.H(cirq.GridQubit(0, 0))])
    c2 = cirq.Circuit([cirq.H(cirq.GridQubit(0, 1))])
    circuit = parallelize_circuits(c1, c2)


def test_safe_tetris_circuits():
    circuit = make_test_circuit()
    circuit = safe_tetris_circuits(circuit, circuit)


def test_safe_concatenate_circuits():
    circuit = make_test_circuit()
    circuit = safe_concatenate_circuits(circuit, circuit)


def test_merge_faster():
    g1 = cirq.rz(np.pi / 4).on(cirq.GridQubit(0, 0))
    g2 = cirq.rz(np.pi / 6).on(cirq.GridQubit(0, 0))
    g3 = merge_faster(g1, g2)

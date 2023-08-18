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

"""Tests for seniority_zero/circuits_simplified/core.py"""

import cirq
import numpy as np

from recirq.seniority_zero.circuits_simplified.core import (
    brickwall_givens_swap_network,
    GHZ_prep_loop_on_lattice,
    lochschmidt_echo,
)


def test_brickwall_givens_swap_network():
    qubits = [cirq.GridQubit(0, j) for j in range(6)]
    params = np.random.uniform(0, 1, 9)
    depth = 3
    circuit = cirq.Circuit(brickwall_givens_swap_network(qubits=qubits, params=params, depth=depth))

    # Test correct depth
    assert len(circuit) == depth

    # Test gates couple to nearest neighbours on loop
    for moment in circuit:
        for gate in moment:
            q1, q2 = gate.qubits
            assert abs(q1.col - q2.col) == 1 or abs(q1.col - q2.col) == 5


def test_GHZ_prep_loop_on_lattice():
    loop = [cirq.GridQubit(0, j) for j in range(3)] + [
        cirq.GridQubit(1, j) for j in reversed(range(3))
    ]
    initial_state = [0, 1, 0, 1, 0, 1]
    circuit = cirq.Circuit(GHZ_prep_loop_on_lattice(loop, initial_state))

    # Test correct depth
    assert len(circuit) == 3


def test_lochschmidt_echo():
    qubits = [cirq.GridQubit(0, j) for j in range(6)]
    params = np.random.uniform(0, 1, 9)
    depth = 3
    circuit = cirq.Circuit(brickwall_givens_swap_network(qubits=qubits, params=params, depth=depth))
    circuit_ls = cirq.Circuit(lochschmidt_echo(circuit, qubits))
    assert len(circuit_ls) == 7
    simulator = cirq.Simulator()
    res = simulator.simulate(circuit_ls)
    assert abs(abs(res.final_state_vector[0]) - 1) < 1e-6

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

"""Tests for seniority_zero.circuits_expt.core.py"""

import cirq
import numpy as np

from recirq.seniority_zero.circuits_expt.core import (
    get_starting_qubits_all_groups_2xN,
    get_starting_qubits_all_groups_loop,
    GHZ_prep_2xn_mixed_filling,
    GHZ_prep_loop_on_lattice,
    givens_swap_network,
    starting_index_GHZ_prep_2xn,
)
from recirq.seniority_zero.scheduling import get_tqbg_groups


def test_givens_swap_network():
    qubits = [cirq.GridQubit(0, j) for j in range(6)]
    params = np.random.uniform(0, 1, 9)
    depth = 3
    circuit = givens_swap_network(qubits, params, depth)
    assert len(circuit) == 19  # 6 gates per layer * 3 layers + initial gate


def test_GHZ_prep_2xn_mixed_filling():
    qubit_line1 = [cirq.GridQubit(0, j) for j in range(3)]
    qubit_line2 = [cirq.GridQubit(1, j) for j in reversed(range(3))]
    state_line1 = [0, 1, 0]
    state_line2 = [1, 0, 1]
    starting_index = starting_index_GHZ_prep_2xn(state_line1, state_line2)
    circuit = GHZ_prep_2xn_mixed_filling(
        qubit_line1, qubit_line2, state_line1, state_line2, starting_index
    )
    assert len(circuit) == 11  # 2 x CNOT + 1 x SWAP compiled from CZ


def test_GHZ_prep_loop_on_lattice():
    qubit_line1 = [cirq.GridQubit(0, j) for j in range(3)]
    qubit_line2 = [cirq.GridQubit(1, j) for j in reversed(range(3))]
    loop = qubit_line1 + qubit_line2
    initial_state = [0, 1, 0, 1, 0, 1]
    circuit = GHZ_prep_loop_on_lattice(loop, initial_state)
    assert len(circuit) == 11


def test_get_starting_qubits_all_groups_2xN():
    qubit_line1 = [cirq.GridQubit(0, j) for j in range(3)]
    qubit_line2 = [cirq.GridQubit(1, j) for j in reversed(range(3))]
    qubits = qubit_line1 + qubit_line2
    groups = get_tqbg_groups(qubits)
    starting_qubits = get_starting_qubits_all_groups_2xN(qubits)
    assert len(starting_qubits) == len(groups)


def test_get_starting_qubits_all_groups_loop():
    qubit_line1 = [cirq.GridQubit(0, j) for j in range(3)]
    qubit_line2 = [cirq.GridQubit(1, j) for j in reversed(range(3))]
    qubits = qubit_line1 + qubit_line2
    groups = get_tqbg_groups(qubits)
    starting_qubits = get_starting_qubits_all_groups_loop(qubits, groups)
    assert len(starting_qubits) == len(groups)

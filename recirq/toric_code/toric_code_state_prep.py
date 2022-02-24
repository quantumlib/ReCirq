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
"""Circuits to generate the ground state in a toric code rectangle

(all plaquettes +1).
"""

from __future__ import annotations

from typing import Iterator, List, Set

import cirq

from . import toric_code_rectangle as tcr


def toric_code_cnot_circuit(
    code: tcr.ToricCodeRectangle, x_basis: bool = False
) -> cirq.Circuit:
    """State prep circuit using CNOTs and Hadamards.

    Args:
        x_basis: If True, we Hadamard all qubits at the end of the circuit, effectively
            changing basis from Z to X, or switching around Z and X plaquettes.
    """
    # Hadamard "captain" qubits
    circuit = cirq.Circuit(cirq.H(q) for q in code.captain_qubits)

    # Work "middle-out" through the columns
    # Each group's "down" CNOTs coexist with the next group's "out" CNOTs
    cnot_moments: List[List[cirq.Operation]] = [[]]
    for idx, cols in enumerate(_middle_out_column_groups(code)):
        cnot_moments.extend([[], []])

        for col in cols:
            for row in range(code.rows):
                captain = code.x_plaquette_to_captain(row, col)
                if idx == 0:  # Center columns: left, right, down
                    q0 = code.q_lower_left(captain)
                    q1 = code.q_lower_right(captain)
                else:  # Outer columns: outside, inside, down
                    q0 = code.q_lower_outside(captain)
                    q1 = code.q_lower_inside(captain)
                q2 = code.q_down(captain)

                cnot_moments[-3].append(cirq.CNOT(captain, q0))
                cnot_moments[-2].append(cirq.CNOT(captain, q1))
                cnot_moments[-1].append(cirq.CNOT(code.q_lower_outside(captain), q2))

    circuit += (cirq.Moment(ops) for ops in cnot_moments)

    # Optionally Hadamard all qubits to switch basis
    if x_basis:
        circuit += cirq.Moment(cirq.H(q) for q in code.qubits)

    return circuit


def _middle_out_column_groups(code: tcr.ToricCodeRectangle) -> Iterator[Set[int]]:
    """Iterate through column indices, starting from the center and working out in pairs."""
    cols = code.cols

    # Start in center: 2 for even cols, 1 for odd cols
    if cols % 2 == 0:
        center_cols = {cols // 2 - 1, cols // 2}
    else:
        center_cols = {cols // 2}
    yield center_cols

    # Work outwards in pairs
    for right_col in range(cols // 2 + 1, cols):
        left_col = cols - right_col - 1
        yield {left_col, right_col}

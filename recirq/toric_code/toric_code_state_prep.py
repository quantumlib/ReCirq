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
"""Module for representing, preparing, and analyzing toric code states.

Rectangular patches of the toric code are represented by ToricCodeRectangle. These are used to
make Cirq circuits to make the toric code ground state with ToricCodeStatePrep. Resulting plaquette
parity expectation values are stored with ToricCodePlaquettes, which can be visualized using
ToricCodePlotter.
"""

from __future__ import annotations

from typing import Iterator, List, Set

import cirq


class ToricCodeStatePrep:
    """Circuits to generate the ground state in a toric code rectangle (all plaquettes +1)."""

    def __init__(self, code: ToricCodeRectangle):
        """

        Args:
            code: Toric code rectangle whose ground state to prepare
        """
        self.code = code

    def __repr__(self) -> str:
        return f"ToricCodeStatePrep(code={self.code})"

    def cnot_circuit(self, x_basis: bool = False) -> cirq.Circuit:
        """State prep circuit using CNOTs and Hadamards.

        Args:
            x_basis: If True, we Hadamard all qubits at the end of the circuit, effectively
                changing basis from Z to X, or switching around Z and X plaquettes.
        """
        # Hadamard "captain" qubits
        circuit = cirq.Circuit(cirq.H(q) for q in self.code.captain_qubits)

        # Work "middle-out" through the columns
        # Each group's "down" CNOTs coexist with the next group's "out" CNOTs
        cnot_moments: List[List[cirq.Operation]] = [[]]
        for idx, cols in enumerate(self._middle_out_column_groups()):
            cnot_moments.extend([[], []])

            for col in cols:
                for row in range(self.code.rows):
                    captain = self.code.x_plaquette_to_captain(row, col)
                    if idx == 0:  # Center columns: left, right, down
                        q0 = self.code.q_lower_left(captain)
                        q1 = self.code.q_lower_right(captain)
                    else:  # Outer columns: outside, inside, down
                        q0 = self.code.q_lower_outside(captain)
                        q1 = self.code.q_lower_inside(captain)
                    q2 = self.code.q_down(captain)

                    cnot_moments[-3].append(cirq.CNOT(captain, q0))
                    cnot_moments[-2].append(cirq.CNOT(captain, q1))
                    cnot_moments[-1].append(
                        cirq.CNOT(self.code.q_lower_outside(captain), q2)
                    )

        circuit += (cirq.Moment(ops) for ops in cnot_moments)

        # Optionally Hadamard all qubits to switch basis
        if x_basis:
            circuit += cirq.Moment(cirq.H(q) for q in self.code.qubits)

        return circuit

    def _middle_out_column_groups(self) -> Iterator[Set[int]]:
        """Iterate through column indices, starting from the center and working out in pairs."""
        cols = self.code.cols

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

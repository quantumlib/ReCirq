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

from typing import Iterator, List, Set, Tuple

import cirq
import numpy as np


def q_displaced(qubit: cirq.GridQubit, displacement: np.ndarray) -> cirq.GridQubit:
    """Helper to return a `GridQubit` displaced by a fixed x/y.

    Args:
        displacement: numpy array with row/column displacement.

    Returns:
        `cirq.GridQubit` that is displaced by a fixed amount of rows/columns/
    """
    return qubit + (int(round(displacement[0])), int(round(displacement[1])))


class ToricCodeRectangle:
    """Abstract representation of a rectangular patch of the toric code on a grid of qubits.

    We think of the toric code as a checkerboard of X and Z plaquettes. In the original Kitaev
    terminology, X plaquettes are "stars" and Z plaquettes are "plaquettes." The rectangle
    consists of an M by N rectangle of X plaquettes. In the bulk, there is a checkerboard of
    X and Z plaquettes, and the boundary is formed by weight-3 and weight-2 Z plaquettes. In
    Kitaev terminology, the rectangle has "rough boundaries." Since all four sides have the same
    boundary condition, there is a unique ground state to the toric code hamiltonian, which gives
    +1 for all of the X and Z plaquettes.

    Here is an example, where blocks of X's and Z's represent X and Z plaquettes, and Q's and C's
    represent the qubits. Each qubit touches two Z plaquettes and one or two X plaquettes.

    ZZZZ C ZZZZZZZ C ZZZZZZZ C ZZZZ
    ZZ  XXX  ZZZ  XXX  ZZZ  XXX  ZZ
    Q XXXXXXX Q XXXXXXX Q XXXXXXX Q
    ZZ  XXX  ZZZ  XXX  ZZZ  XXX  ZZ
    ZZZZ C ZZZZZZZ C ZZZZZZZ C ZZZZ
    ZZ  XXX  ZZZ  XXX  ZZZ  XXX  ZZ
    Q XXXXXXX Q XXXXXXX Q XXXXXXX Q
    ZZ  XXX  ZZZ  XXX  ZZZ  XXX  ZZ
    ZZZZ Q ZZZZZZZ Q ZZZZZZZ Q ZZZZ

    Each X plaquette has a "captain" qubit (C) associated with it, on the upper corner. These play
    a special role in the quantum circuit used to generate the toric code ground state.

    In this example, there are two rows and three columns of X plaquettes, with the upper-left
    indexed (row=0, col=0). The Z plaquettes are indexed similarly, where (row, col) refers to the Z
    plaquette to the upper-left of the (row, col) X plaquette, so (row=0, col=0) is the triangular
    weight-2 Z plaquette in the upper-left corner.

    There is a second coordinate system in place, GridQubit indices, as in a physical qubit layout
    on a rectangular array. This system is rotated with respect to the toric code plaquette indices.
    Four orientations are possible. Consider the same 17 qubits in this orientation:

      c0↓  c1↓  c2↓  c3↓  c4↓
    r0→      Q    C

    r1→ Q    C    Q    C

    r2→ Q    Q    C    Q    C

    r3→      Q    Q    C    Q

    r4→           Q    Q

    In this example, the "captain" qubits are (0, 2), (1, 1), (1, 3), (2, 2), (2, 4), and (3, 3).
    We define the orientation by the "row vector" from the (row=0, col=0) "captain" qubit (0, 2)
    to the (row=1, col=0) "captain" qubit (1, 1) in GridQubit coordinates, which is (1, -1) here.

    In particular, this would be represented by
    ToricCodeRectangle(
        origin_qubit=cirq.GridQubit(0, 2),
        row_vector=(1, -1),
        rows=2,
        cols=3,
    )
    """

    def __init__(
        self,
        origin_qubit: cirq.GridQubit,
        row_vector: Tuple[int, int],
        rows: int,
        cols: int,
    ):
        """

        Args:
            origin_qubit: "Captain" (top) qubit of the upper-left X plaquette, in the toric code
                reference frame
            row_vector: Displacement between origin_qubit and the opposite qubit in its X plaquette,
                which is the leftmost "captain" qubit in the second row of X plaquettes. This
                determines the orientation of the toric code on the qubit grid. Must be (±1, ±1).
            rows: Number of rows of X plaquettes
            cols: Number of columns of X plaquettes
        """
        if np.any(np.abs(row_vector) != np.array([1, 1])):
            raise ValueError(f"Illegal row_vector={row_vector}: must be (±1, ±1)")

        if rows < 1 or cols < 1:
            raise ValueError(f"rows ({rows}) and cols ({cols}) must both be at least 1")

        self.origin_qubit = origin_qubit
        self.row_vector = np.array(row_vector, dtype=int)
        self.col_vector = np.array((-row_vector[1], row_vector[0]), dtype=int)
        self.rows = rows
        self.cols = cols
        self.qubits_reading_order = self._qubits_reading_order()

    def __repr__(self) -> str:
        return (
            f"ToricCodeRectangle(origin_qubit=cirq.GridQubit{self.origin_qubit}, "
            f"row_vector={tuple(self.row_vector)}, rows={self.rows}, cols={self.cols})"
        )

    @property
    def captain_qubits(self) -> Set[cirq.GridQubit]:
        return {
            self.x_plaquette_to_captain(row, col)
            for row, col in self.x_plaquette_indices()
        }

    @property
    def qubits(self) -> Set[cirq.GridQubit]:
        return {
            qubit
            for row, col in self.x_plaquette_indices()
            for qubit in self.x_plaquette_to_qubits(row, col)
        }

    def _qubits_reading_order(self) -> Tuple[cirq.GridQubit, ...]:
        """Qubits listed in "reading order."

        Returns:
            A tuple with the qubits enumerated starting with the "captain"
            of x_plaquette (0, 0) and then reading right
            (the first self.cols qubits are the captains of row 0)
            and going down row by row.
        """
        qubits: List[cirq.GridQubit] = []
        for row in range(self.rows):
            # Get top row (captains)
            for col in range(self.cols):
                qubits.append(self.x_plaquette_to_captain(row, col))
            # Get middle row
            for col in range(self.cols + 1):
                captain = self.x_plaquette_to_captain(row, col)
                qubits.append(self.q_lower_left(captain))

        # Get bottom row (phantom captains)
        for col in range(self.cols):
            qubits.append(self.x_plaquette_to_captain(self.rows, col))

        return tuple(qubits)

    def x_plaquette_indices(self) -> Iterator[Tuple[int, int]]:
        """Iterator over (row, col) index pairs for x plaquettes."""
        for row in range(self.rows):
            for col in range(self.cols):
                yield row, col

    def z_plaquette_indices(self) -> Iterator[Tuple[int, int]]:
        """Iterator over (row, col) index pairs for z plaquettes."""
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                yield row, col

    def is_corner(self, row: int, col: int) -> bool:
        """True if the z plaquette at (row, col) is a corner."""
        return (row, col) in {
            (0, 0),
            (0, self.cols),
            (self.rows, 0),
            (self.rows, self.cols),
        }

    def is_edge(self, row: int, col: int) -> bool:
        """True if the z plaquette at (row, col) is on an edge (non-corner)."""
        if self.is_corner(row, col):
            return False
        return row == 0 or row == self.rows or col == 0 or col == self.cols

    def is_boundary(self, row: int, col: int) -> bool:
        """True if the z plaquette at (row, col) is an edge or corner."""
        return self.is_corner(row, col) or self.is_edge(row, col)

    def x_plaquette_to_captain(self, row: int, col: int) -> cirq.GridQubit:
        displacement = row * self.row_vector + col * self.col_vector
        return q_displaced(self.origin_qubit, displacement)

    def x_plaquette_to_qubits(self, row: int, col: int) -> Set[cirq.GridQubit]:
        captain = self.x_plaquette_to_captain(row, col)
        return {
            captain,
            self.q_lower_left(captain),
            self.q_lower_right(captain),
            self.q_down(captain),
        }

    def z_plaquette_to_qubits(self, row: int, col: int) -> Set[cirq.GridQubit]:
        right_qubit = self.x_plaquette_to_captain(row, col)
        possible_qubits = {
            right_qubit,
            self.q_upper_left(right_qubit),
            self.q_across_left(right_qubit),
            self.q_lower_left(right_qubit),
        }
        return possible_qubits & self.qubits

    def x_plaquette_to_qubit_idxs(self, row: int, col: int) -> Set[int]:
        plaquette_qubits = self.x_plaquette_to_qubits(row, col)
        all_qubits = sorted(self.qubits)
        return {all_qubits.index(q) for q in plaquette_qubits}

    def z_plaquette_to_qubit_idxs(self, row: int, col: int) -> Set[int]:
        plaquette_qubits = self.z_plaquette_to_qubits(row, col)
        all_qubits = sorted(self.qubits)
        return {all_qubits.index(q) for q in plaquette_qubits}

    def captain_to_x_plaquette(self, captain_qubit: cirq.GridQubit) -> Tuple[int, int]:
        dr_grid = captain_qubit.row - self.origin_qubit.row
        dc_grid = captain_qubit.col - self.origin_qubit.col

        c0, c1 = self.col_vector
        r0, r1 = self.row_vector

        plaq_row = (c0 * dc_grid - c1 * dr_grid) / (c0 * r1 - c1 * r0)
        plaq_col = (r1 * dr_grid - r0 * dc_grid) / (c0 * r1 - c1 * r0)
        plaq_vector = np.array((plaq_row, plaq_col))

        if not np.allclose(plaq_vector, np.round(plaq_vector)):
            raise ValueError(
                f'{captain_qubit} is not a valid "captain" qubit for '
                f"origin {self.origin_qubit}, row_vector {self.row_vector}."
            )

        return int(round(plaq_row)), int(round(plaq_col))

    def q_lower_left(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        return q_displaced(captain_qubit, (self.row_vector - self.col_vector) / 2)

    def q_lower_right(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        return q_displaced(captain_qubit, (self.row_vector + self.col_vector) / 2)

    def q_down(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        return q_displaced(captain_qubit, self.row_vector)

    def q_upper_left(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        return q_displaced(captain_qubit, (-self.row_vector - self.col_vector) / 2)

    def q_across_left(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        return q_displaced(captain_qubit, -self.col_vector)

    def q_lower_outside(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        _row, col = self.captain_to_x_plaquette(captain_qubit)
        if col < self.cols // 2:
            return self.q_lower_left(captain_qubit)
        return self.q_lower_right(captain_qubit)

    def q_lower_inside(self, captain_qubit: cirq.GridQubit) -> cirq.GridQubit:
        _row, col = self.captain_to_x_plaquette(captain_qubit)
        if col < self.cols // 2:
            return self.q_lower_right(captain_qubit)
        return self.q_lower_left(captain_qubit)

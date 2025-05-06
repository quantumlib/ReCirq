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

from collections.abc import Iterator
from enum import Enum

import cirq
import numpy as np

class QubitNeighbor(Enum):
    """Enum for the neighbors of a qubit in a grid.

    The neighbors are defined as follows:
    - A: The qubit itself
    - U: Up
    - R: Right
    - D: Down
    - L: Left
    """
    A = "a"
    U = "u"
    R = "r"
    D = "d"
    L = "l"

class LGTGrid:
    """Representation of rectangular grid of qubits for lattice gauge theory experiments.

    This grid of qubits is analagous to the X-type/Z-type Toric Code with the boundary chosen
    such that all boundary stabilizers are Z-type. The origin_qubit is chosen to be an ancillary
    qubit for an X-type plaquette closest to a corner. The other qubits are chosen based on the
    orientation vector, which is drawn from the origin_qubit away from the corner to the adjacent
    'bulk' ancillary qubit for a Z-type plaquette.
    """

    def __init__(
        self,
        origin_qubit: cirq.GridQubit,
        orientation_vector: tuple[int, int],
        rows: int,
        cols: int,
        flip_rowcol: bool = False,
    ):
        """

        Args:
            origin_qubit: ancillary qubit for an X-type plaquette closest to a corner
            orientation_vector: vector drawn from origin_qubit to the ancillary qubit of the adjacent 'bulk'
                Z-type plaquette
            rows: number of rows of X-type stabilizers
            cols: number of columns of X-type stabilizers
            flip_rowcol: chooses whether to transpose row and columns code
        """
        if np.any(np.abs(orientation_vector) != np.array([1, 1])):
            raise ValueError(
                f"Inconsistent orientation_vector={orientation_vector}: must be (±1, ±1)"
            )

        if rows < 1 or cols < 1:
            raise ValueError(f"rows ({rows}) and cols ({cols}) must both be at least 1")

        self.origin_qubit = origin_qubit
        self.orientation_vector = orientation_vector
        self.row_vector = np.array([2 * orientation_vector[0], 0], dtype=int)
        self.col_vector = np.array([0, 2 * orientation_vector[1]], dtype=int)
        self.rows = rows
        self.cols = cols
        self.flip_rowcol = flip_rowcol

    @property
    def physical_qubits(self) -> set[cirq.GridQubit]:
        """Set of physical gauge qubits at the corners of Z and X plaquettes."""
        return sorted(
            self.u_set(self.x_ancillary_qubits)
            .union(self.r_set(self.x_ancillary_qubits))
            .union(self.d_set(self.x_ancillary_qubits))
            .union(self.l_set(self.x_ancillary_qubits))
        )
    
    @property
    def x_ancillary_qubits(self) -> set[cirq.GridQubit]:
        return {self.x_plaquette_to_x_ancilla(row, col) for row, col in self.x_plaquette_indices}
    
    @property
    def z_ancillary_l_side_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(row, 0) for row in range(0, self.rows + 1)}
    
    @property
    def z_ancillary_dl_corner_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(self.rows, 0)}
    
    @property
    def z_even_col_ancillary_qubits(self) -> set[cirq.GridQubit]:
        return {
            self.z_plaquette_to_z_ancilla(row, col)
            for row, col in self.z_plaquette_even_col_indices
        }
    
    @property
    def z_ancillary_d_side_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(self.rows, col) for col in range(0, self.cols + 1)}
    
    @property
    def z_ancillary_ul_corner_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(0, 0)}
    
    @property
    def z_ancillary_u_side_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(0, col) for col in range(0, self.cols + 1)}
    
    @property
    def z_ancillary_r_side_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(row, self.cols) for row in range(0, self.rows + 1)}

    @property
    def z_ancillary_ur_corner_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(0, self.cols)}
    
    @property
    def z_odd_col_ancillary_qubits(self) -> set[cirq.GridQubit]:
        return {
            self.z_plaquette_to_z_ancilla(row, col) for row, col in self.z_plaquette_odd_col_indices
        }
    
    @property
    def x_ancillary_qubits_by_col(self) -> list[set[cirq.GridQubit]]:
        return [
            {self.x_plaquette_to_x_ancilla(row, col) for row in range(self.rows)}
            for col in range(self.cols)
        ]
    
    @property
    def z_ancillary_dr_corner_qubits(self) -> set[cirq.GridQubit]:
        return {self.z_plaquette_to_z_ancilla(self.rows, self.cols)}
    
    @property
    def x_even_col_ancillary_qubits(self) -> set[cirq.GridQubit]:
        return {
            self.x_plaquette_to_x_ancilla(row, col)
            for row, col in self.x_plaquette_even_col_indices
        }
    
    @property
    def x_odd_col_ancillary_qubits(self) -> set[cirq.GridQubit]:
        return {
            self.x_plaquette_to_x_ancilla(row, col) for row, col in self.x_plaquette_odd_col_indices
        }
    
    @property
    def x_plaquette_indices(self) -> Iterator[tuple[int, int]]:
        for row in range(self.rows):
            for col in range(self.cols):
                yield row, col

    @property
    def z_plaquette_indices(self) -> Iterator[tuple[int, int]]:
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                yield row, col

    @property
    def z_plaquette_even_col_indices(self) -> Iterator[tuple[int, int]]:
        for row in range(self.rows + 1):
            for col in range(0, self.cols + 1, 2):
                yield row, col

    @property
    def z_plaquette_odd_col_indices(self) -> Iterator[tuple[int, int]]:
        for row in range(self.rows + 1):
            for col in range(1, self.cols + 1, 2):
                yield row, col

    @property
    def x_plaquette_even_col_indices(self) -> Iterator[tuple[int, int]]:
        for row in range(self.rows):
            for col in range(0, self.cols, 2):
                yield row, col

    @property
    def x_plaquette_odd_col_indices(self) -> Iterator[tuple[int, int]]:
        for row in range(self.rows):
            for col in range(1, self.cols, 2):
                yield row, col
    
    def q_displaced(self, qubit: cirq.GridQubit, displacement: np.ndarray) -> cirq.GridQubit:
        """Helper to return a `GridQubit` displaced by a fixed x/y.

        Args:
            displacement: numpy array with row/column displacement.

        Returns:
            `cirq.GridQubit` that is displaced by a fixed amount of rows/columns
        """
        if self.flip_rowcol is False:
            return qubit + (int(round(displacement[0])), int(round(displacement[1])))
        else:
            return qubit + (int(round(displacement[1])), int(round(displacement[0])))
        
    def u_set(self, qubit_set: set[cirq.GridQubit]) -> set[cirq.GridQubit]:
        """Ouputs a set of qubits translated up from the input set."""
        return {self.q_displaced(qubit, -self.row_vector / 2) for qubit in qubit_set}

    def r_set(self, qubit_set: set[cirq.GridQubit]) -> set[cirq.GridQubit]:
        """Ouputs a set of qubits translated right from the input set."""
        return {self.q_displaced(qubit, self.col_vector / 2) for qubit in qubit_set}

    def d_set(self, qubit_set: set[cirq.GridQubit]) -> set[cirq.GridQubit]:
        """Ouputs a set of qubits translated down from the input set."""
        return {self.q_displaced(qubit, self.row_vector / 2) for qubit in qubit_set}

    def l_set(self, qubit_set: set[cirq.GridQubit]) -> set[cirq.GridQubit]:
        """Ouputs a set of qubits translated left from the input set."""
        return {self.q_displaced(qubit, -self.col_vector / 2) for qubit in qubit_set}
    
    def ancillary_to_pair(
        self, a_set: set[cirq.GridQubit], qubit1_relationship: QubitNeighbor, qubit2_relationship: QubitNeighbor
    ) -> set[tuple[cirq.GridQubit, cirq.GridQubit]]:
        """Generates a pair of qubits for each qubit in a_set based on nearest neighbor
        relationships.

        Args:
            a_set: set of input qubits, usually ancillary qubits of a set of plaquettes.
            qubit1_relationship: either A, U, R, D, or L. indicated that the first qubit in each pair will either
                be qubits in a_set (A), or the qubits that are up (U), right (R), down (D), or left (L) from
                the qubits in a_set.
            qubit2_relationship: ... same but the second qubit in each pair.
        """
        displacement_dict ={
            QubitNeighbor.A: 0,
            QubitNeighbor.U: -self.row_vector / 2,
            QubitNeighbor.R: self.col_vector / 2,
            QubitNeighbor.D: self.row_vector / 2,
            QubitNeighbor.L: -self.col_vector / 2,

        }

        return {
            (
                self.q_displaced(qubit, displacement_dict[qubit1_relationship]),
                self.q_displaced(qubit, displacement_dict[qubit2_relationship])
            )
            for qubit in a_set
        }
    
    
    def z_plaquette_to_physical_qubit_indices(self, row: int, col: int) -> list[int]:
        """Outputs the indices of all physical qubits within the sorted list of all physical
        qubits."""
        return [
            sorted(list(self.physical_qubits)).index(qubit)
            for qubit in self.z_plaquette_to_physical_qubits(row, col).values()
        ]
    
    def x_plaquette_to_physical_qubit_indices(self, row: int, col: int) -> list[int]:
        """Outputs the indices of all physical qubits within the sorted list of all physical
        qubits."""
        return [
            sorted(list(self.physical_qubits)).index(qubit)
            for qubit in self.x_plaquette_to_physical_qubits(row, col).values()
        ]
    
    def z_plaquette_to_physical_qubits(self, row: int, col: int) -> dict[str : cirq.GridQubit]:
        """Outputs all the physical qubits that constitute the Z plaquette at indices (row, col).

        Interior Z plaquettes are made up of 4 qubits. While plaquettes on the edge are only comprised
        of 3 qubits. Corner plaquettes just have 2 qubits.
        """
        u_displacement = (
            (row - 0.5) * self.row_vector + (col - 0.5) * self.col_vector - self.row_vector / 2
        )
        r_displacement = (
            (row - 0.5) * self.row_vector + (col - 0.5) * self.col_vector + self.col_vector / 2
        )
        d_displacement = (
            (row - 0.5) * self.row_vector + (col - 0.5) * self.col_vector + self.row_vector / 2
        )
        l_displacement = (
            (row - 0.5) * self.row_vector + (col - 0.5) * self.col_vector - self.col_vector / 2
        )
        if row == 0 and col == 0:
            return {
                "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
                "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
            }
        elif row == 0 and col == self.cols:
            return {
                "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
                "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
            }
        elif row == self.rows and col == 0:
            return {
                "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
                "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
            }
        elif row == self.rows and col == self.cols:
            return {
                "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
                "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
            }
        elif row == 0:
            return {
                "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
                "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
                "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
            }
        elif col == 0:
            return {
                "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
                "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
                "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
            }
        elif row == self.rows:
            return {
                "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
                "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
                "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
            }
        elif col == self.cols:
            return {
                "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
                "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
                "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
            }
        else:
            return {
                "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
                "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
                "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
                "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
            }
        
    def x_plaquette_to_physical_qubits(self, row: int, col: int) -> dict[str : cirq.GridQubit]:
        """Outputs all the physical qubits that constitute the X plaquette at indices (row, col).

        In current construction, all X plaquettes are on the interior, so thus have 4 qubits.
        """
        u_displacement = (row) * self.row_vector + (col) * self.col_vector - self.row_vector / 2
        r_displacement = (row) * self.row_vector + (col) * self.col_vector + self.col_vector / 2
        d_displacement = (row) * self.row_vector + (col) * self.col_vector + self.row_vector / 2
        l_displacement = (row) * self.row_vector + (col) * self.col_vector - self.col_vector / 2
        return {
            "u_qubit": self.q_displaced(self.origin_qubit, u_displacement),
            "r_qubit": self.q_displaced(self.origin_qubit, r_displacement),
            "d_qubit": self.q_displaced(self.origin_qubit, d_displacement),
            "l_qubit": self.q_displaced(self.origin_qubit, l_displacement),
        }

    def x_plaquette_to_x_ancilla(self, row: int, col: int) -> cirq.GridQubit:
        """This displacement is counting from the upper left X plaquette.

        Args:
            row: plaquette row, row 0 is on top with increasing rows going down.
            col: plaquette column, column 0 is on the left and increasing to the left.
        """
        displacement = row * self.row_vector + col * self.col_vector
        return self.q_displaced(self.origin_qubit, displacement)
    
    def z_plaquette_to_z_ancilla(self, row: int, col: int) -> cirq.GridQubit:
        """This displacement is counting from the upper left Z plaquette.

        Args:
            row: plaquette row, row 0 is on top with increasing rows going down.
            col: plaquette column, column 0 is on the left and increasing to the left.
        """
        displacement = (row - 1 / 2) * self.row_vector + (col - 1 / 2) * self.col_vector
        return self.q_displaced(self.origin_qubit, displacement)
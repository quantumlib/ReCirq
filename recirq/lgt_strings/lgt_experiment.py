from collections.abc import Iterator, Sequence
from typing import Any

import cirq
from matplotlib import colormaps
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import sympy

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
        
    def u_set(self, qubit_set: set[cirq.GridQubit]):
        """Ouputs a set of qubits translated up from the input set."""
        return {self.q_displaced(qubit, -self.row_vector / 2) for qubit in qubit_set}

    def r_set(self, qubit_set: set[cirq.GridQubit]):
        """Ouputs a set of qubits translated right from the input set."""
        return {self.q_displaced(qubit, self.col_vector / 2) for qubit in qubit_set}

    def d_set(self, qubit_set: set[cirq.GridQubit]):
        """Ouputs a set of qubits translated down from the input set."""
        return {self.q_displaced(qubit, self.row_vector / 2) for qubit in qubit_set}

    def l_set(self, qubit_set: set[cirq.GridQubit]):
        """Ouputs a set of qubits translated left from the input set."""
        return {self.q_displaced(qubit, -self.col_vector / 2) for qubit in qubit_set}
    
    def ancillary_to_pair(
        self, a_set: set[cirq.GridQubit], qubit1_relationship: str, qubit2_relationship: str
    ) -> set[tuple[cirq.GridQubit, cirq.GridQubit]]:
        """Generates a pair of qubits for each qubit in a_set based on nearest neighbor
        relationships.

        Args:
            a_set: set of input qubits, usually ancillary qubits of a set of plaquettes.
            qubit1_relationship: either "a", "u", "r", "d", or "l". indicated that the first qubit in each pair will either
                be qubits in a_set ("a"), or the qubits that are up ("u"), right ("r"), down ("d"), or left ("l") from
                the qubits in a_set.
            qubit2_relationship: ... same but the second qubit in each pair.
        """
        if qubit1_relationship == "a":
            if qubit2_relationship == "u":
                return {(qubit, self.q_displaced(qubit, -self.row_vector / 2)) for qubit in a_set}
            elif qubit2_relationship == "r":
                return {(qubit, self.q_displaced(qubit, self.col_vector / 2)) for qubit in a_set}
            elif qubit2_relationship == "d":
                return {(qubit, self.q_displaced(qubit, self.row_vector / 2)) for qubit in a_set}
            elif qubit2_relationship == "l":
                return {(qubit, self.q_displaced(qubit, -self.col_vector / 2)) for qubit in a_set}
            else:
                raise ValueError(
                    f"error in input values {qubit1_relationship} or {qubit2_relationship}"
                )
        elif qubit1_relationship == "u":
            if qubit2_relationship == "a":
                return {(self.q_displaced(qubit, -self.row_vector / 2), qubit) for qubit in a_set}
            elif qubit2_relationship == "r":
                return {
                    (
                        self.q_displaced(qubit, -self.row_vector / 2),
                        self.q_displaced(qubit, self.col_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "d":
                return {
                    (
                        self.q_displaced(qubit, -self.row_vector / 2),
                        self.q_displaced(qubit, self.row_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "l":
                return {
                    (
                        self.q_displaced(qubit, -self.row_vector / 2),
                        self.q_displaced(qubit, -self.col_vector / 2),
                    )
                    for qubit in a_set
                }
            else:
                raise ValueError(
                    f"error in input values {qubit1_relationship} or {qubit2_relationship}"
                )
        elif qubit1_relationship == "r":
            if qubit2_relationship == "a":
                return {(self.q_displaced(qubit, self.col_vector / 2), qubit) for qubit in a_set}
            elif qubit2_relationship == "u":
                return {
                    (
                        self.q_displaced(qubit, self.col_vector / 2),
                        self.q_displaced(qubit, -self.row_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "d":
                return {
                    (
                        self.q_displaced(qubit, self.col_vector / 2),
                        self.q_displaced(qubit, self.row_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "l":
                return {
                    (
                        self.q_displaced(qubit, self.col_vector / 2),
                        self.q_displaced(qubit, -self.col_vector / 2),
                    )
                    for qubit in a_set
                }
            else:
                raise ValueError(
                    f"error in input values {qubit1_relationship} or {qubit2_relationship}"
                )
        elif qubit1_relationship == "d":
            if qubit2_relationship == "a":
                return {(self.q_displaced(qubit, self.row_vector / 2), qubit) for qubit in a_set}
            elif qubit2_relationship == "u":
                return {
                    (
                        self.q_displaced(qubit, self.row_vector / 2),
                        self.q_displaced(qubit, -self.row_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "r":
                return {
                    (
                        self.q_displaced(qubit, self.row_vector / 2),
                        self.q_displaced(qubit, self.col_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "l":
                return {
                    (
                        self.q_displaced(qubit, self.row_vector / 2),
                        self.q_displaced(qubit, -self.col_vector / 2),
                    )
                    for qubit in a_set
                }
            else:
                raise ValueError(
                    f"error in input values {qubit1_relationship} or {qubit2_relationship}"
                )
        elif qubit1_relationship == "l":
            if qubit2_relationship == "a":
                return {(self.q_displaced(qubit, -self.col_vector / 2), qubit) for qubit in a_set}
            elif qubit2_relationship == "u":
                return {
                    (
                        self.q_displaced(qubit, -self.col_vector / 2),
                        self.q_displaced(qubit, -self.row_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "r":
                return {
                    (
                        self.q_displaced(qubit, -self.col_vector / 2),
                        self.q_displaced(qubit, self.col_vector / 2),
                    )
                    for qubit in a_set
                }
            elif qubit2_relationship == "d":
                return {
                    (
                        self.q_displaced(qubit, -self.col_vector / 2),
                        self.q_displaced(qubit, self.row_vector / 2),
                    )
                    for qubit in a_set
                }
            else:
                raise ValueError(
                    f"error in input values {qubit1_relationship} or {qubit2_relationship}"
                )
        else:
            raise ValueError(
                f"error in input values {qubit1_relationship} or {qubit2_relationship}"
            )
    
    def z_plaquette_to_physical_qubit_indices(self, row: int, col: int):
        """Outputs the indices of all physical qubits within the sorted list of all physical
        qubits."""
        return [
            sorted(list(self.physical_qubits)).index(qubit)
            for qubit in self.z_plaquette_to_physical_qubits(row, col).values()
        ]
    
    def x_plaquette_to_physical_qubit_indices(self, row: int, col: int):
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

def variational_ground_state_minimal_qubits_cols(
    grid: LGTGrid, x_ancillary_qubits_in_cols: list[set[cirq.GridQubit]], theta: float
) -> list[cirq.Moment]:
    """Moments to prepare the state from the toric code variational ansatz for two columns.

    Instead of applying a unitary on each plaquette that creates an equal superpostion of all spins up
    and all spins down (the Hadamard), this creates a weighted superposition of
    cos(theta) |0000> + sin(theta) |1111>, based on the variational parameter theta.

    Args:
        grid: grid of qubits to be used
        x_ancillary_qubits_in_cols: list of sets of x-plaquette ancillary qubits. The zeroth entry should
            be the ancillary qubits of the left column of x-plaquettes and the first entry should be for
            the right column.
        theta: dictionary where the key:value is he:theta
    """

    x_ancillary_qubits_all = x_ancillary_qubits_in_cols[0].union(x_ancillary_qubits_in_cols[1])

    return [
        cirq.Moment(cirq.Ry(rads=theta).on_each(grid.u_set(x_ancillary_qubits_all))),
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(x_ancillary_qubits_all, "u", "l")),
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(x_ancillary_qubits_all, "u", "r")),
        *cirq.Circuit.zip(
            cirq.Circuit.from_moments(
                *cnot_on_layer(
                    pairs_list=grid.ancillary_to_pair(x_ancillary_qubits_in_cols[0], "l", "d")
                )
            ),
            cirq.Circuit.from_moments(
                *cnot_on_layer(
                    pairs_list=grid.ancillary_to_pair(x_ancillary_qubits_in_cols[1], "r", "d")
                )
            ),
        ).moments,
    ]


def variational_ground_state_minimal_qubits(
    grid: LGTGrid, theta: float, extra_x_plaquette_indices: list[tuple[int, int]] = []
) -> list[cirq.Moment]:
    """Moments to prepare the state from the toric code variation ansatz.

    Instead of applying a unitary on each plaquette that creates an equal superpostion of all spins up
    and all spins down (the Hadamard), this creates a weighted superposition of
    cos(theta) |0000> + sin(theta) |1111>, based on the variation parameter theta.

    Args:
        grid: grid of qubits to be used
        theta: dictionary where the key:value is he:theta
        extra_x_plaquette_indices: list of (row,col) of extra x plaquettes to prepare the ground state for.
    """
    extra_ancilla_qubits = {}
    for col in range(grid.cols):
        extra_ancilla_qubits[col] = []
        for p in extra_x_plaquette_indices:
            if p[1] == col:
                extra_ancilla_qubits[col].append(grid.x_plaquette_to_x_ancilla(p[0], p[1]))

    moments = []

    if (grid.cols / 2).is_integer():
        for col in range(int(grid.cols / 2), grid.cols):
            moments = moments + variational_ground_state_minimal_qubits_cols(
                grid,
                [
                    grid.x_ancillary_qubits_by_col[grid.cols - col - 1].union(
                        extra_ancilla_qubits[grid.cols - col - 1]
                    ),
                    grid.x_ancillary_qubits_by_col[col].union(extra_ancilla_qubits[col]),
                ],
                theta=theta,
            )
    else:
        moments = moments + variational_ground_state_minimal_qubits_cols(
            grid,
            [
                grid.x_ancillary_qubits_by_col[int(grid.cols / 2 - 0.5)].union(
                    extra_ancilla_qubits[int(grid.cols / 2 - 0.5)]
                ),
                {},
            ],
            theta=theta,
        )
        for col in range(1, int(grid.cols / 2 + 0.5)):
            moments = moments + variational_ground_state_minimal_qubits_cols(
                grid,
                [
                    grid.x_ancillary_qubits_by_col[int(grid.cols / 2 - 0.5 - col)].union(
                        extra_ancilla_qubits[int(grid.cols / 2 - 0.5 - col)]
                    ),
                    grid.x_ancillary_qubits_by_col[int(grid.cols / 2 - 0.5 + col)].union(
                        extra_ancilla_qubits[int(grid.cols / 2 - 0.5 + col)]
                    ),
                ],
                theta=theta,
            )

    return moments

def trotter_even_zcol_entangle_minimal_qubits(
    grid: LGTGrid, extra_z_plaquette_indices: list[tuple[int, int]] = []
):
    """Minimal qubits need not obey device connectivity, used for simulation.

    Args:
        extra_plaquette_indices: Can specify extra plaquettes to include in the Trotterization.
            By default all 4 qubits around the extra plaquettes will be included.
    """
    extra_even_qubits = []
    for p in extra_z_plaquette_indices:
        if p[1] % 2 == 0:
            extra_even_qubits.append(grid.z_plaquette_to_z_ancilla(p[0], p[1]))
    extra_z_plaquette_ancillary_qubit = set(extra_even_qubits)

    return [
        *cnot_on_layer(
            pairs_list=grid.ancillary_to_pair(
                grid.z_ancillary_l_side_qubits
                - grid.z_ancillary_dl_corner_qubits
                - extra_z_plaquette_ancillary_qubit,
                "d",
                "r",
            ).union(
                grid.ancillary_to_pair(
                    (
                        grid.z_even_col_ancillary_qubits
                        - grid.z_ancillary_l_side_qubits
                        - grid.z_ancillary_d_side_qubits
                    ).union(extra_z_plaquette_ancillary_qubit),
                    "d",
                    "l",
                )
            )
        ),
        *cnot_on_layer(
            pairs_list=grid.ancillary_to_pair(
                grid.z_ancillary_l_side_qubits
                - grid.z_ancillary_ul_corner_qubits
                - grid.z_ancillary_dl_corner_qubits
                - extra_z_plaquette_ancillary_qubit,
                "r",
                "u",
            )
            .union(
                grid.ancillary_to_pair(
                    grid.z_ancillary_dl_corner_qubits - extra_z_plaquette_ancillary_qubit, "r", "u"
                )
            )
            .union(
                grid.ancillary_to_pair(
                    (
                        grid.z_even_col_ancillary_qubits
                        - grid.z_ancillary_l_side_qubits
                        - grid.z_ancillary_u_side_qubits
                    ).union(extra_z_plaquette_ancillary_qubit),
                    "l",
                    "u",
                )
            )
            .union(
                grid.ancillary_to_pair(
                    grid.z_ancillary_u_side_qubits.intersection(grid.z_even_col_ancillary_qubits)
                    - grid.z_ancillary_l_side_qubits
                    - grid.z_ancillary_r_side_qubits
                    - extra_z_plaquette_ancillary_qubit,
                    "l",
                    "r",
                )
            )
        ),
        *cnot_on_layer(
            pairs_list=grid.ancillary_to_pair(
                (
                    grid.z_even_col_ancillary_qubits
                    - grid.z_ancillary_l_side_qubits
                    - grid.z_ancillary_u_side_qubits
                    - grid.z_ancillary_r_side_qubits
                ).union(extra_z_plaquette_ancillary_qubit),
                "r",
                "u",
            )
        ),
    ]


def trotter_odd_zcol_entangle_minimal_qubits(
    grid: LGTGrid, extra_z_plaquette_indices: list[tuple[int, int]] = []
):
    """Minimal qubits need not obey device connectivity, used for simulation.

    Args:
        extra_plaquette_indices: Can specify extra plaquettes to include in the Trotterization.
            By default all 4 qubits around the extra plaquettes will be included.
    """
    extra_odd_qubits = []
    for p in extra_z_plaquette_indices:
        if p[1] % 2 == 1:
            extra_odd_qubits.append(grid.z_plaquette_to_z_ancilla(p[0], p[1]))
    extra_z_plaquette_ancillary_qubit = set(extra_odd_qubits)

    return [
        *cnot_on_layer(
            pairs_list=grid.ancillary_to_pair(
                (grid.z_odd_col_ancillary_qubits - grid.z_ancillary_d_side_qubits).union(
                    extra_z_plaquette_ancillary_qubit
                ),
                "d",
                "l",
            )
        ),
        *cnot_on_layer(
            pairs_list=grid.ancillary_to_pair(
                (
                    grid.z_odd_col_ancillary_qubits
                    - grid.z_ancillary_u_side_qubits
                    - grid.z_ancillary_dr_corner_qubits
                ).union(extra_z_plaquette_ancillary_qubit),
                "l",
                "u",
            )
            .union(
                grid.ancillary_to_pair(
                    grid.z_ancillary_u_side_qubits.intersection(grid.z_odd_col_ancillary_qubits)
                    - grid.z_ancillary_r_side_qubits
                    - extra_z_plaquette_ancillary_qubit,
                    "l",
                    "r",
                )
            )
            .union(
                grid.ancillary_to_pair(
                    grid.z_ancillary_dr_corner_qubits.intersection(grid.z_odd_col_ancillary_qubits)
                    - extra_z_plaquette_ancillary_qubit,
                    "u",
                    "l",
                )
            )
        ),
        *cnot_on_layer(
            pairs_list=grid.ancillary_to_pair(
                (
                    grid.z_odd_col_ancillary_qubits
                    - grid.z_ancillary_u_side_qubits
                    - grid.z_ancillary_r_side_qubits
                ).union(extra_z_plaquette_ancillary_qubit),
                "r",
                "u",
            )
        ),
    ]

def trotter_even_xcol_entangle_minimal_qubits(
    grid: LGTGrid, extra_plaquette_indices: list[tuple[int, int]] = []
):
    """Minimal qubits need not obey device connectivity, used for simulation.

    Args:
        extra_plaquette_indices: Can specify extra plaquettes to include in the Trotterization.
            By default all 4 qubits around the extra plaquettes will be included.
    """

    extra_even_plaquette_indices = []
    for p in extra_plaquette_indices:
        if p[1] % 2 == 0:
            extra_even_plaquette_indices.append(p)

    ancilla_qubits = grid.x_even_col_ancillary_qubits.union(
        {grid.x_plaquette_to_x_ancilla(p[0], p[1]) for p in extra_even_plaquette_indices}
    )

    return [
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(ancilla_qubits, "d", "l")),
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(ancilla_qubits, "l", "u")),
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(ancilla_qubits, "r", "u")),
    ]


def trotter_odd_xcol_entangle_minimal_qubits(
    grid: LGTGrid, extra_plaquette_indices: list[tuple[int, int]] = []
):
    """Minimal qubits need not obey device connectivity, used for simulation.

    Args:
        extra_plaquette_indices: Can specify extra plaquettes to include in the Trotterization.
            By default all 4 qubits around the extra plaquettes will be included.
    """

    extra_odd_plaquette_indices = []
    for p in extra_plaquette_indices:
        if p[1] % 2 == 1:
            extra_odd_plaquette_indices.append(p)

    ancilla_qubits = grid.x_odd_col_ancillary_qubits.union(
        {grid.x_plaquette_to_x_ancilla(p[0], p[1]) for p in extra_odd_plaquette_indices}
    )

    return [
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(ancilla_qubits, "d", "l")),
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(ancilla_qubits, "l", "u")),
        *cnot_on_layer(pairs_list=grid.ancillary_to_pair(ancilla_qubits, "r", "u")),
    ]

def trotter_step_minimal_qubits(
    grid: LGTGrid,
    dt: sympy.Symbol,
    coupling: sympy.Symbol,
    he: sympy.Symbol,
    Je: sympy.Symbol = 1,
    Jm: sympy.Symbol = 1,
    extra_z_plaquette_indices: list[tuple[int, int]] = [],
    extra_x_plaquette_indices: list[tuple[int, int]] = [],
):
    """Generate moments to simulate a full Trotter step of the lattice gauge theory Hamiltonian.

    Hamiltonian can be seen at, for example, https://arxiv.org/abs/0912.3272 equation (1.2)
    We set Je = Jm = 1 and hx = -coupling and hz = -he.

    Args:
        grid: Representation of set of qubits.
        dt: trotter time step.
        coupling: sigma x field strength.
        he: sigma z field strength.
        Je: vertex strength
        Jm: plaquette strength
    """
    extra_z_plaquette_ancillary_qubits = {
        grid.z_plaquette_to_z_ancilla(plaquette_indices[0], plaquette_indices[1])
        for plaquette_indices in extra_z_plaquette_indices
    }

    extra_x_plaquette_ancillary_qubits = {
        grid.x_plaquette_to_x_ancilla(p[0], p[1]) for p in extra_x_plaquette_indices
    }
    extra_x_plaquette_even_col_ancillary_qubits = set([])
    extra_x_plaquette_odd_col_ancillary_qubits = set([])

    for p in extra_x_plaquette_indices:
        if p[1] % 2 == 0:
            extra_x_plaquette_even_col_ancillary_qubits.add(
                grid.x_plaquette_to_x_ancilla(p[0], p[1])
            )
        else:
            extra_x_plaquette_odd_col_ancillary_qubits.add(
                grid.x_plaquette_to_x_ancilla(p[0], p[1])
            )

    extended_physical_qubits = sorted(
        grid.u_set(grid.x_ancillary_qubits.union(extra_x_plaquette_ancillary_qubits))
        .union(grid.r_set(grid.x_ancillary_qubits.union(extra_x_plaquette_ancillary_qubits)))
        .union(grid.d_set(grid.x_ancillary_qubits.union(extra_x_plaquette_ancillary_qubits)))
        .union(grid.l_set(grid.x_ancillary_qubits.union(extra_x_plaquette_ancillary_qubits)))
    )

    return [
        *trotter_even_zcol_entangle_minimal_qubits(grid, extra_z_plaquette_indices),
        cirq.Moment(
            cirq.rz(-2 * Je * dt).on_each(
                grid.u_set(
                    (grid.z_even_col_ancillary_qubits - grid.z_ancillary_u_side_qubits).union(
                        extra_z_plaquette_ancillary_qubits.intersection(
                            grid.z_even_col_ancillary_qubits
                        )
                    )
                )
            ),
            cirq.rz(-2 * Je * dt).on_each(
                grid.r_set(
                    grid.z_ancillary_u_side_qubits.intersection(grid.z_even_col_ancillary_qubits)
                    - grid.z_ancillary_ur_corner_qubits
                    - extra_z_plaquette_ancillary_qubits
                )
            ),
            cirq.rz(-2 * Je * dt).on_each(
                grid.l_set(
                    grid.z_ancillary_ur_corner_qubits.intersection(grid.z_even_col_ancillary_qubits)
                    - extra_z_plaquette_ancillary_qubits
                )
            ),
        ),
        *trotter_even_zcol_entangle_minimal_qubits(grid, extra_z_plaquette_indices)[::-1],
        *trotter_odd_zcol_entangle_minimal_qubits(grid, extra_z_plaquette_indices),
        cirq.Moment(
            cirq.rz(-2 * Je * dt).on_each(
                grid.u_set(
                    (
                        grid.z_odd_col_ancillary_qubits
                        - grid.z_ancillary_u_side_qubits
                        - grid.z_ancillary_dr_corner_qubits
                    ).union(
                        extra_z_plaquette_ancillary_qubits.intersection(
                            grid.z_odd_col_ancillary_qubits
                        )
                    )
                )
            ),
            cirq.rz(-2 * Je * dt).on_each(
                grid.r_set(
                    grid.z_ancillary_u_side_qubits.intersection(grid.z_odd_col_ancillary_qubits)
                    - grid.z_ancillary_r_side_qubits
                    - extra_z_plaquette_ancillary_qubits
                )
            ),
            cirq.rz(-2 * Je * dt).on_each(
                grid.l_set(
                    grid.z_ancillary_ur_corner_qubits.union(
                        grid.z_ancillary_dr_corner_qubits
                    ).intersection(grid.z_odd_col_ancillary_qubits)
                    - extra_z_plaquette_ancillary_qubits
                )
            ),
        ),
        *trotter_odd_zcol_entangle_minimal_qubits(grid, extra_z_plaquette_indices)[::-1],
        cirq.Moment(cirq.H.on_each(extended_physical_qubits)),
        *trotter_even_xcol_entangle_minimal_qubits(
            grid, extra_plaquette_indices=extra_x_plaquette_indices
        ),
        cirq.Moment(
            cirq.rz(-2 * Jm * dt).on_each(
                grid.u_set(
                    grid.x_even_col_ancillary_qubits.union(
                        extra_x_plaquette_even_col_ancillary_qubits
                    )
                )
            )
        ),
        *trotter_even_xcol_entangle_minimal_qubits(
            grid, extra_plaquette_indices=extra_x_plaquette_indices
        )[::-1],
        *trotter_odd_xcol_entangle_minimal_qubits(
            grid, extra_plaquette_indices=extra_x_plaquette_indices
        ),
        cirq.Moment(
            cirq.rz(-2 * Jm * dt).on_each(
                grid.u_set(
                    grid.x_odd_col_ancillary_qubits.union(
                        extra_x_plaquette_odd_col_ancillary_qubits
                    )
                )
            )
        ),
        *trotter_odd_xcol_entangle_minimal_qubits(
            grid, extra_plaquette_indices=extra_x_plaquette_indices
        )[::-1],
        cirq.Moment(cirq.H.on_each(extended_physical_qubits)),
        cirq.Moment(cirq.rz(-2 * he * dt).on_each(grid.physical_qubits)),
        cirq.Moment(cirq.rx(-2 * coupling * dt).on_each(grid.physical_qubits)),
    ]

def plaquette_bitstrings(
    data: np.array, grid: LGTGrid, particle_locs: list[tuple[int, int]] = []
) -> np.ndarray:
    """Converts Z basis qubit bitstring to a bitstring in the Z stabilizers of the toric code.

    Outputs the bitstring for the Z stabilizers where 0 corresponds to <ZZZZ>=1
    and -1 to <ZZZZ>=-1.

    Args:
        data: array where the 0 axis indexes the shot and the 1 axis indexes the
            qubits.
        grid: representation of set of qubits.
        particle_locs: Flips the value of the plaquette bitstrings for the plaquettes indicated
            here, for applications with the particle_inject Hamiltonian quench.
    """
    plaquette_bitstrings = np.zeros((np.shape(data)[0], (grid.rows + 1) * (grid.cols + 1)))
    for row in range(grid.rows + 1):
        for col in range(grid.cols + 1):
            if (row, col) not in particle_locs:
                p = col + (grid.cols + 1) * row
                plaquette_bitstrings[:, p] = (
                    np.sum(
                        data[
                            :,
                            list(
                                grid.z_plaquette_to_physical_qubit_indices(
                                    p // (grid.cols + 1), p % (grid.cols + 1)
                                )
                            ),
                        ],
                        axis=1,
                    )
                    % 2
                )
            else:
                p = col + (grid.cols + 1) * row
                plaquette_bitstrings[:, p] = (
                    np.sum(
                        data[
                            :,
                            list(
                                grid.z_plaquette_to_physical_qubit_indices(
                                    p // (grid.cols + 1), p % (grid.cols + 1)
                                )
                            ),
                        ],
                        axis=1,
                    )
                    + 1
                ) % 2

    return plaquette_bitstrings


def x_plaquette_bitstrings(data: np.array, grid: LGTGrid) -> np.ndarray:
    """Coverts X basis qubit bitstring to a bitstring  in the X stabilizers of the toric code.

    Outputes the bitstring of the X stabilizers where 0 corresponds to <XXXX>=1
    and 1 to <XXXX>=-1

    Args:
        data: array where the 0 axis indexes the shot and the 1 axis indexes the
            qubits.
        grid: representation of set of qubits.
    """
    plaquette_bitstrings = np.zeros((np.shape(data)[0], (grid.rows) * (grid.cols)))
    for row in range(grid.rows):
        for col in range(grid.cols):
            p = col + (grid.cols) * row
            plaquette_bitstrings[:, p] = (
                np.sum(
                    data[
                        :,
                        list(
                            grid.x_plaquette_to_physical_qubit_indices(
                                p // (grid.cols), p % (grid.cols)
                            )
                        ),
                    ],
                    axis=1,
                )
                % 2
            )

    return plaquette_bitstrings

def cnot_on_layer(
    pairs_list: Sequence[tuple[cirq.GridQubit, cirq.GridQubit]],
    depolarization_probability: float | dict | None = None,
) -> Sequence[cirq.Moment]:
    """Outputs a list of moments for CNOT between two lists, in terms of CZ gates.

    Args:
        pairs_list: list of pairs of qubits with the first qubit in each pair being the control qubit and the second
            qubit being the target
        depolarization_probability: parameter to a 2 qubit depolarization channel after the CZ gate. This may also be a
            dictionary that maps specific depolarization probability to specific pairs of qubits.
    """
    if depolarization_probability is None:
        return [
            cirq.Moment(cirq.H.on_each(pair[1] for pair in pairs_list)),
            cirq.Moment(cirq.CZ.on(qc, qt) for qc, qt in pairs_list),
            cirq.Moment(cirq.H.on_each(pair[1] for pair in pairs_list)),
        ]
    elif type(depolarization_probability) == float:
        return [
            cirq.Moment(cirq.H.on_each(pair[1] for pair in pairs_list)),
            cirq.Moment(cirq.CZ.on(qc, qt) for qc, qt in pairs_list),
            cirq.Moment(
                cirq.depolarize(depolarization_probability, 2).on(pair[0], pair[1])
                for pair in pairs_list
            ),
            cirq.Moment(cirq.H.on_each(pair[1] for pair in pairs_list)),
        ]
    elif type(depolarization_probability) == dict:
        # make sure there is a error for every pair.
        assert set(pairs_list).intersection(set(depolarization_probability.keys())) == set(
            pairs_list
        )

        return [
            cirq.Moment(cirq.H.on_each(pair[1] for pair in pairs_list)),
            cirq.Moment(cirq.CZ.on(qc, qt) for qc, qt in pairs_list),
            cirq.Moment(
                cirq.depolarize(depolarization_probability[pair], 2).on(pair[0], pair[1])
                for pair in pairs_list
            ),
            cirq.Moment(cirq.H.on_each(pair[1] for pair in pairs_list)),
        ]
    
def excitation_sep_plaquette_input(
    plaq_data: np.ndarray, plaq_rows: int = 3, plaq_cols: int = 2
) -> np.ndarray:
    mat1 = np.zeros((plaq_rows * plaq_cols, plaq_rows * plaq_cols))
    mat1[:] = np.arange(np.shape(mat1)[1]) % plaq_cols
    mat1 = np.transpose(mat1)

    mat2 = np.zeros((plaq_rows * plaq_cols, plaq_rows * plaq_cols))
    mat2[:] = np.arange(np.shape(mat2)[1]) % plaq_cols

    mat3 = np.zeros((plaq_rows * plaq_cols, plaq_rows * plaq_cols))
    mat3[:] = np.arange(np.shape(mat3)[1]) // plaq_cols
    mat3 = np.transpose(mat3)

    mat4 = np.zeros((plaq_rows * plaq_cols, plaq_rows * plaq_cols))
    mat4[:] = np.arange(np.shape(mat4)[1]) // plaq_cols

    distance_metric = (abs(mat1 - mat2) + abs(mat3 - mat4)) / 2

    distances = np.einsum("ij,jl,il->i", plaq_data, distance_metric, plaq_data)

    return distances

def plot_qubit_polarization_values(
    grid:LGTGrid,
    qubit_polarization_data: np.ndarray,
    ancilla_states_data:np.ndarray,
    ax: plt.Axes | None = None,
    qubit_kwargs: dict[str, Any] | None = None,
    set_axis_off: bool = True,
    force_equal_aspect: bool = True,
    qubit_colormap=colormaps.get_cmap("Oranges"),
    ancilla_colormap = colormaps.get_cmap("Oranges"),
    plot_physical_qubits: bool | list[list[tuple[int]]] = False,
    plot_ancillas: bool | list[list[tuple[int]]] = False,
) -> plt.Axes:
    """Plot toric code plaquette expectation values as colored tiles.
    If round_edges = True, boundary Z-plaquettes are rounded off with
    edges corresponding to stabilizers, otherwise all four quibts of
    the boundary plaquettes are shown."""

    if qubit_kwargs is None:
        qubit_kwargs = {"edgecolor": "k", "linewidth": 1.0}

    rows = grid.rows
    cols = grid.cols

    # Set up axis
    if ax is None:
        _fig, ax = plt.subplots()
    if set_axis_off:  # Remove axis frame
        ax.set_axis_off()
    if force_equal_aspect:
        ax.set_aspect("equal")
    ax.set_xlim(-1, cols)
    ax.set_ylim(rows, -1.5)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    qubit_order = []
    for row in range(grid.rows+1):
        for col in range(grid.cols):
            qubit_order.append(*grid.u_set({grid.x_plaquette_to_x_ancilla(row,col)}))
    for row in range(1,grid.rows+1):
        for col in range(grid.cols+1):
            qubit_order.append(*grid.u_set({grid.z_plaquette_to_z_ancilla(row,col)}))
    qubit_order

    pols_shuffled = np.array([qubit_polarization_data[np.nonzero(np.array(grid.physical_qubits) == qubit)[0][0]] for qubit in qubit_order])

    if plot_physical_qubits is True:
        # Plot X plaquette top qubits
        qubit_index = 0
        for row in range(grid.rows + 1):
            for col in range(grid.cols):
                ax.add_patch(
                    get_qubit_patch_rect(
                        pols_shuffled,
                        qubit_index,
                        row,
                        col,
                        x_basis=True,
                        qubit_colormap=qubit_colormap,
                        **qubit_kwargs,
                    )
                )

                qubit_index += 1

        # Plot Z plaquette bottom qubits
        for row, col in grid.z_plaquette_indices:
            if row != grid.rows:
                ax.add_patch(
                    get_qubit_patch_rect(
                        pols_shuffled,
                        qubit_index,
                        row,
                        col,
                        x_basis=False,
                        qubit_colormap=qubit_colormap,
                        **qubit_kwargs,
                    )
                )

                qubit_index += 1

    elif type(plot_physical_qubits) == list:
        # Plot X plaquette top qubits
        for row, col in plot_physical_qubits[0]:
            qubit_index = (row) * grid.cols + col
            ax.add_patch(
                get_qubit_patch_rect(
                    qubit_polarization_data,
                    qubit_index,
                    row,
                    col,
                    x_basis=True,
                    qubit_colormap=qubit_colormap,
                    **qubit_kwargs,
                )
            )

        # Plot Z plaquette bottom qubits
        for row, col in plot_physical_qubits[1]:
            qubit_index = (
                (data.code.rows + 1) * (data.code.cols + 1) + (row) * (data.code.cols + 1) + col
            )

            ax.add_patch(
                get_qubit_patch_rect(
                    qubit_polarization_data,
                    qubit_index,
                    row,
                    col,
                    x_basis=False,
                    round_edges=round_edges,
                    qubit_colormap=qubit_colormap,
                    **qubit_kwargs,
                )
            )
            
    if plot_ancillas is True:
        for row, col in grid.z_plaquette_indices:
            qubit_index = (2*grid.cols+1)*row+col
            ax.add_patch(
                get_ancilla_patch(data = ancilla_states_data, qubit_index = qubit_index,row=row, col=col, x_basis=False, ancilla_cmap = ancilla_colormap, **qubit_kwargs)
            )
        for row, col in grid.x_plaquette_indices:
            qubit_index = (2*grid.cols+1)*row+3+col
            ax.add_patch(
                get_ancilla_patch(data = ancilla_states_data, qubit_index = qubit_index,row=row, col=col, x_basis=True, ancilla_cmap = ancilla_colormap, **qubit_kwargs)
                )

    elif type(plot_ancillas) == list:
        for row, col in plot_ancillas[0]:
            ax.add_patch(
                get_ancilla_patch(row=row, col=col, x_basis=False, **qubit_kwargs)
            )
        for row, col in plot_ancillas[1]:
            ax.add_patch(get_ancilla_patch(row=row, col=col, x_basis=True, **qubit_kwargs))
    return ax

def get_qubit_patch_rect(
    data: np.array,
    qubit_index: int,
    row: int,
    col: int,
    x_basis: bool,
    qubit_colormap=colormaps.get_cmap("Oranges"),
    **kwargs,
) -> mpatches.Patch:
    """Generate a single patch polygon for a plaquette."""
    if not x_basis:  # z basis, includes special boundary cases
        coordinates = [
            (col - 0.5 + 0.2, row),
            (col - 0.5, row + 0.25),
            (col - 0.5 - 0.2, row),
            (col - 0.5, row - 0.25),
        ]

    else:  # x basis
        coordinates = [
            (col + 0.25, row - 0.5),
            (col, row - 0.5 + 0.2),
            (col - 0.25, row - 0.5),
            (col, row - 0.5 - 0.2),
        ]

    color = qubit_colormap((data[qubit_index] - 1)/(-2))

    return mpatches.Polygon(coordinates, closed=True, facecolor=color, **kwargs)

def get_ancilla_patch(
        data:np.ndarray,
        qubit_index,
        row: int,
        col: int,
        x_basis: bool,
        ancilla_cmap,
        **kwargs
    ) -> mpatches.Patch:
    """Generate a single patch polygon for a plaquette."""
    if not x_basis:
        coordinates = (col - 0.5, row - 0.5)

    else:  # x basis
        coordinates = (col, row)

    color = ancilla_cmap(data[qubit_index])

    return mpatches.Circle(coordinates, radius=0.146, facecolor=color, **kwargs)
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
from typing import Dict, Iterable, Tuple

import pandas as pd

from . import toric_code_rectangle as tcr


class ToricCodePlaquettes:
    """X and Z plaquette (stabilizer) expectation values."""

    def __init__(
        self,
        code: tcr.ToricCodeRectangle,
        x_plaquettes: Dict[Tuple[int, int], float],
        z_plaquettes: Dict[Tuple[int, int], float],
    ):
        """

        Args:
            code: Toric code rectangle whose plaquette expectation values we store
            x_plaquettes: Mapping from X plaquette index (row, col) to expectation value.
                We expect rows in range(code.rows) and cols in range(code.cols).
            z_plaquettes: Mapping from Z plaquette index (row, col) to expectation value.
                We expect rows in range(code.rows + 1) and cols in range(code.cols + 1).
        """
        self.code = code
        self.x_plaquettes = x_plaquettes
        self.z_plaquettes = z_plaquettes

    def __repr__(self) -> str:
        return (
            f"ToricCodePlaquettes(code={self.code}, "
            f"x_plaquettes={self.x_plaquettes}, z_plaquettes={self.z_plaquettes})"
        )

    @classmethod
    def for_uniform_parity(
        cls, code: tcr.ToricCodeRectangle, x_value: float, z_value: float
    ) -> "ToricCodePlaquettes":
        """Create plaquettes with uniform X and Z expectation values."""
        x_plaquettes = {(row, col): x_value for row, col in code.x_plaquette_indices()}
        z_plaquettes = {(row, col): z_value for row, col in code.z_plaquette_indices()}
        return cls(code, x_plaquettes, z_plaquettes)

    @classmethod
    def from_global_measurements(
        cls, code: tcr.ToricCodeRectangle, x_data: pd.DataFrame, z_data: pd.DataFrame
    ) -> "ToricCodePlaquettes":
        """Compute stabilizer expectation values from global measurement data.

        Args:
            code: Toric code rectangle whose plaquette expectation values we store
            x_data: Result of global measurements in the X basis. DataFrame with a single column of
                integer values, one for each measurement outcome, as in cirq.Result.data.
            z_data: Result of global measurements in the Z basis, similar to x_data.

        Returns:
            ToricCodePlaquettes with expectation values calculated from data
        """
        x_plaquettes: Dict[Tuple[int, int], float] = {
            (row, col): cls.expectation_value(code, x_data, row, col, x_basis=True)
            for row, col in code.x_plaquette_indices()
        }
        z_plaquettes: Dict[Tuple[int, int], float] = {
            (row, col): cls.expectation_value(code, z_data, row, col, x_basis=False)
            for row, col in code.z_plaquette_indices()
        }
        return cls(code, x_plaquettes, z_plaquettes)

    @classmethod
    def expectation_value(
        cls,
        code: tcr.ToricCodeRectangle,
        data: pd.DataFrame,
        row: int,
        col: int,
        x_basis: bool,
    ) -> float:
        """Compute an expectation value for a single X or Z plaquette.

        Args:
            code: Toric code rectangle with this plaquette
            data: DataFrame with a single column of integer values, one for each measurement
                outcome, as in cirq.Result.data
            row: Plaquette row
            col: Plaquette column
            x_basis: If True, look at the X plaquette at (row, col); otherwise, look at the Z
                plaquette at (row, col)

        Returns:
            Expectation value of the plaquette, between -1 and 1
        """
        if x_basis:
            qubit_idxs = code.x_plaquette_to_qubit_idxs(row, col)
        else:
            qubit_idxs = code.z_plaquette_to_qubit_idxs(row, col)

        total_qubits = len(code.qubits)
        parities = data.applymap(
            lambda value: cls.compute_parity(value, qubit_idxs, total_qubits)
        )
        return float(parities.mean())

    @staticmethod
    def compute_parity(value: int, qubit_idxs: Iterable[int], total_qubits: int) -> int:
        """Compute the parity of a set of qubits for a given measurement.

        Args:
            value: Big-endian packed integer of qubit measurement outcomes
            qubit_idxs: Select the measurement outcomes for qubits at these indices
            total_qubits: Total number of qubits. This is needed to know how many bits to use when
                expanding value in binary; it determines where idx=0 is.

        Returns:
            +1 for even parity (even number of 1s), -1 for odd parity (odd number of 1s)
        """
        bitstring = f"{value:0{total_qubits}b}"
        number_of_ones = sum(int(bitstring[idx]) for idx in qubit_idxs)
        return (-1) ** number_of_ones

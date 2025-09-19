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

import cirq

from recirq.lattice_gauge.lattice_gauge_grid import LGTGrid

def test_z_plaquette_to_physical_qubit_indices():
    # Create a mock LGTGrid
    grid = LGTGrid(
        origin_qubit=cirq.GridQubit(0, 0),
        orientation_vector=(1, 1),
        rows=3,
        cols=3
    )

    # Define a sample plaquette index
    plaquette_index = (1, 1)

    # Call the function to test
    physical_qubit_indices = grid.z_plaquette_to_physical_qubit_indices(row = plaquette_index[0], col = plaquette_index[1])

    # Expected result based on the grid structure
    expected_indices = [4,8,11,7]

    # Assert the result matches the expected indices
    assert physical_qubit_indices == expected_indices, (
        f"Expected {expected_indices}, but got {physical_qubit_indices}"
    )

def test_z_plaquette_to_physical_qubits():
    # Create a mock LGTGrid
    grid = LGTGrid(
        origin_qubit=cirq.GridQubit(0, 0),
        orientation_vector=(1, 1),
        rows=3,
        cols=3
    )

    # Define a sample plaquette index
    plaquette_index = (1, 1)

    # Call the function to test
    physical_qubits = grid.z_plaquette_to_physical_qubits(row=plaquette_index[0], col=plaquette_index[1])

    # Expected result based on the grid structure
    expected_qubits = {
        'u_qubit': cirq.GridQubit(0, 1),
         'r_qubit': cirq.GridQubit(1, 2),
         'd_qubit': cirq.GridQubit(2, 1),
         'l_qubit': cirq.GridQubit(1, 0)
    }

    # Assert the result matches the expected qubits
    assert physical_qubits == expected_qubits, (
        f"Expected {expected_qubits}, but got {physical_qubits}"
    )
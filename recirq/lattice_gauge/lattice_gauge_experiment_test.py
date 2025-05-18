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

import numpy as np
import matplotlib.pyplot as plt
from recirq.lattice_gauge.lattice_gauge_experiment import plot_qubit_polarization_values, variational_ground_state_minimal_qubits, trotter_step_minimal_qubits
from recirq.lattice_gauge.lattice_gauge_grid import LGTGrid

def test_plot_qubit_polarization_values():
    # Create a mock LGTGrid
    grid = LGTGrid(
        origin_qubit=cirq.GridQubit(0, 0),
        orientation_vector=(1, 1),
        rows=3,
        cols=3
    )

    # Mock data for qubit polarization and ancilla states
    qubit_polarization_data = np.random.rand(len(grid.physical_qubits))
    ancilla_states_data = np.random.rand(len(grid.physical_qubits))

    # Create a matplotlib axis
    fig, ax = plt.subplots()

    # Call the function to test
    plot_qubit_polarization_values(
        grid=grid,
        qubit_polarization_data=qubit_polarization_data,
        ancilla_states_data=ancilla_states_data,
        ax=ax,
        plot_physical_qubits=True
    )

    # Assert that the axis has been modified
    assert ax.has_data(), "The axis should have data after plotting."

    # Close the plot to avoid resource warnings
    plt.close(fig)

def test_variational_ground_state_minimal_qubits():
    # Define grid parameters
    lx, ly = 4, 3  # Grid dimensions
    grid = LGTGrid(
        origin_qubit=cirq.GridQubit(0, 0),
        orientation_vector=(1, 1),
        rows=lx - 1,
        cols=ly - 1,
        flip_rowcol=False
    )

    # Define Hamiltonian coefficients
    hamiltonian_coefs = {'Je': np.random.random(), 'Jm': np.random.random(), 'he':np.random.random(), 'lambda': np.random.random()}

    # Define thetas to test
    thetas = [0, np.pi / 2]

    # Initialize simulator
    simulator = cirq.Simulator()

    # Loop over thetas and compute energy
    # Note the correspondence between he and theta in this test is not physical
    for theta in thetas:
        # Create the circuit for the given theta
        circuit = cirq.Circuit.from_moments(
            *variational_ground_state_minimal_qubits(grid, theta)
        )
        
        observable = cirq.PauliSum()
        for row in range(lx):
            for col in range(ly):
                observable += cirq.PauliString(hamiltonian_coefs['Je'],cirq.Z.on_each(grid.z_plaquette_to_physical_qubits(row, col).values()))
        for row in range(lx-1):
            for col in range(ly-1):
                observable += cirq.PauliString(hamiltonian_coefs['Jm'],cirq.X.on_each(grid.x_plaquette_to_physical_qubits(row, col).values()))
        for qubit in grid.physical_qubits:
            observable += cirq.PauliString(hamiltonian_coefs['he'],cirq.Z(qubit))
            observable += cirq.PauliString(hamiltonian_coefs['lambda'],cirq.X(qubit))

        # Simulate the expectation values
        results = simulator.simulate_expectation_values(circuit,[observable])

        if theta == np.pi:
            assert np.isclose(results[0], lx*ly*hamiltonian_coefs['Je']+(lx-1)*(ly-1)*hamiltonian_coefs['Jm'], atol=1e-2), (
            f"Error of energy of WALA initial state when theta = {theta}"
            )
        elif theta == 0:
            assert np.isclose(results[0], lx*ly*hamiltonian_coefs['Je']+len(grid.physical_qubits)*hamiltonian_coefs['he'], atol=1e-2), (
            f"Error of energy of WALA initial state when theta = {theta}"
            )

def test_trotter_step_minimal_qubits():
    # Define grid parameters
    lx, ly = 3, 2  # Grid dimensions
    grid = LGTGrid(
        origin_qubit=cirq.GridQubit(0, 0),
        orientation_vector=(1, 1),
        rows=lx - 1,
        cols=ly - 1,
        flip_rowcol=False
    )

    # Define Hamiltonian coefficients
    hamiltonian_coefs = {'Je': 1, 'Jm': 1, 'he':0.4, 'lambda': 0.5}
    dt = 0.3

    observable = cirq.PauliSum()

    #Going to test that, under these parameters, the average probability of particle creation from the
    #WALA state after 20 Trotter steps is consistent with the expected value of 0.9019627769788106.
    for row in range(lx):
        for col in range(ly):
            observable += cirq.PauliString(1/6,cirq.Z.on_each(grid.z_plaquette_to_physical_qubits(row, col).values()))

    circuit = cirq.Circuit.from_moments(
        *variational_ground_state_minimal_qubits(grid, 0.625),
        *trotter_step_minimal_qubits(grid, dt, hamiltonian_coefs['lambda'], hamiltonian_coefs['he'], hamiltonian_coefs['Je'], hamiltonian_coefs['Jm'])*20,
    )

    simulator = cirq.Simulator()
    results = simulator.simulate_expectation_values(circuit,[observable])

    assert np.isclose(results[0], (0.9019627769788106+0j), atol=1e-4), ("Error in Trotterization circuit.")

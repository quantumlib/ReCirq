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

"""Generates the Trotterized circuits for 1D Disorder-Free Localization (DFL) experiments.
"""

from collections.abc import Sequence
from typing import List

import cirq
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

#import Enums
from recirq.dfl.dfl_enums import TwoQubitGate, InitialState, Basis

def trotter_circuit(
    grid: Sequence[cirq.GridQubit],
    n_cycles: int,
    dt: float,
    h: float,
    mu: float,
    two_qubit_gate: TwoQubitGate = TwoQubitGate.CZ,
) -> cirq.Circuit:
    """Constructs a Trotter circuit for 1D Disorder-Free Localization (DFL) simulation.

    Args:
        grid: The 1D sequence of qubits used in the experiment.
        n_cycles: The number of Trotter steps/cycles to include.
        dt: The time step size for the Trotterization.
        h: The gauge field strength coefficient.
        mu: The matter field strength coefficient.
        two_qubit_gate: The type of two-qubit gate to use in the layers.
            Use an Enum member from TwoQubitGate (CZ or CPHASE).

    Returns:
        The complete cirq.Circuit for the Trotter evolution.

    Raises:
        ValueError: If an invalid `two_qubit_gate` option is provided.
    """
    if two_qubit_gate.value == TwoQubitGate.CZ.value:
        return _layer_floquet_cz(grid, dt, h, mu) * n_cycles

    elif two_qubit_gate.value == TwoQubitGate.CPHASE.value:
        if n_cycles == 0:
            return cirq.Circuit()
        if n_cycles == 1:
            return _layer_floquet_cphase_first(
                grid, dt, h, mu
            ) + _layer_floquet_cphase_last_missing_piece(grid, dt, h, mu)
        else:
            return (
                _layer_floquet_cphase_first(grid, dt, h, mu)
                + (n_cycles - 1) * _layer_floquet_cphase_middle(grid, dt, h, mu)
                + _layer_floquet_cphase_last_missing_piece(grid, dt, h, mu)
            )
    else:
        raise ValueError("Two-qubit gate can only be cz or cphase")


def _layer_floquet_cz(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    moment_rz = []
    moment_rx = []
    moment_h = []
    for i in range(len(grid)):
        q = grid[i]
        if i % 2 == 1:
            moment_rz.append(cirq.rz(dt).on(q))
            moment_h.append(cirq.H(q))
            moment_rx.append(cirq.rx(2 * h * dt).on(q))
        else:
            moment_rx.append(cirq.rx(2 * mu * dt).on(q))
    return (
        cirq.Circuit.from_moments(cirq.Moment(moment_rz))
        + _change_basis(grid)
        + cirq.Circuit.from_moments(cirq.Moment(moment_rx))
        + _change_basis(grid)
        + cirq.Circuit.from_moments(cirq.Moment(moment_rz))
    )


def _layer_floquet_cphase_middle(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    n = len(grid)
    moment_0 = []
    moment_1 = []
    moment_h = []
    moment_rz = []
    moment_rx = []

    for i in range(n // 2):
        q0 = grid[(2 * i)]
        q = grid[2 * i + 1]
        q1 = grid[(2 * i + 2) % n]
        moment_0.append(cirq.CZ(q0, q))  # left to right
        moment_1.append(cirq.cphase(-4 * dt).on(q1, q))  # right to left

        moment_h.append(cirq.H(q))

        moment_rz.append(cirq.rz(2 * dt).on(q))
        moment_rz.append(cirq.rz(2 * dt).on(q1))

    for i in range(n):
        q = grid[i]
        if i % 2 == 1:
            moment_rx.append(cirq.rx(2 * h * dt).on(q))
        else:
            moment_rx.append(cirq.rx(2 * mu * dt).on(q))

    moment_rz_new = []
    qubits_covered = []
    for gate in moment_rz:
        count = moment_rz.count(gate)
        qubit = gate.qubits[0]
        if qubit not in qubits_covered:
            moment_rz_new.append(cirq.rz(2 * count * dt).on(qubit))
            qubits_covered.append(qubit)

    return cirq.Circuit.from_moments(
        cirq.Moment(moment_h),
        cirq.Moment(moment_0),
        cirq.Moment(moment_h),
        cirq.Moment(moment_rz_new),
        cirq.Moment(moment_1),
        cirq.Moment(moment_h),
        cirq.Moment(moment_0),
        cirq.Moment(moment_h),
        cirq.Moment(moment_rx),
    )


def _layer_floquet_cphase_first(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    moment_rz = []
    moment_rx = []

    for i in range(len(grid)):
        q = grid[i]
        if i % 2 == 1:
            moment_rz.append(cirq.rz(dt).on(q))
            moment_rx.append(cirq.rx(2 * h * dt).on(q))
        else:
            moment_rx.append(cirq.rx(2 * mu * dt).on(q))

    return (
        cirq.Circuit.from_moments(cirq.Moment(moment_rz))
        + _change_basis(grid)
        + cirq.Circuit.from_moments(cirq.Moment(moment_rx))
    )


def _layer_floquet_cphase_last_missing_piece(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    moment_rz = []
    for i in range(len(grid)):
        q = grid[i]
        if i % 2 == 1:
            moment_rz.append(cirq.rz(dt).on(q))

    return _change_basis(grid) + cirq.Circuit.from_moments(cirq.Moment(moment_rz))


def _layer_hadamard(grid: Sequence[cirq.GridQubit], which_qubits="all") -> cirq.Circuit:
    moment = []
    for i in range(len(grid)):
        q = grid[i]

        if i % 2 == 1 and (which_qubits == "gauge" or which_qubits == "all"):
            moment.append(cirq.H(q))

        elif i % 2 == 0 and (which_qubits == "matter" or which_qubits == "all"):
            moment.append(cirq.H(q))
    return cirq.Circuit.from_moments(cirq.Moment(moment))


def _layer_measure(grid: Sequence[cirq.GridQubit]) -> cirq.Circuit:
    moment = []
    for i in range(len(grid) // 2):
        moment.append(cirq.measure(grid[2 * i], key="m"))
    return cirq.Circuit.from_moments(cirq.Moment(moment))


def _change_basis(grid: Sequence[cirq.GridQubit]) -> cirq.Circuit:
    """change the basis so that the matter sites encode the gauge charge."""
    n = len(grid)
    moment_1 = []
    moment_2 = []
    moment_h = []
    for i in range(0, n // 2):
        q1 = grid[(2 * i)]
        q2 = grid[2 * i + 1]
        q3 = grid[(2 * i + 2) % n]
        moment_1.append(cirq.CZ(q1, q2))
        moment_2.append(cirq.CZ(q3, q2))
        moment_h.append(cirq.H(q2))
    return cirq.Circuit.from_moments(
        cirq.Moment(moment_h),
        cirq.Moment(moment_1),
        cirq.Moment(moment_2),
        cirq.Moment(moment_h),
    )


def _energy_bump_initial_state(
    grid, h: float, matter_config: InitialState, excited_qubits: Sequence[cirq.GridQubit]
) -> cirq.Circuit:
    """Circuit for energy bump initial state."""

    theta = np.arctan(h)
    moment = []
    for i in range(len(grid)):
        q = grid[i]
        if i % 2 == 1:
            if q in excited_qubits:
                moment.append(cirq.ry(np.pi + theta).on(q))
            else:
                moment.append(cirq.ry(theta).on(q))

        else:
            if matter_config.value == InitialState.SINGLE_SECTOR.value:
                moment.append(cirq.H(q))
    return cirq.Circuit.from_moments(moment)


def get_1d_dfl_experiment_circuits(
    grid: Sequence[cirq.GridQubit],
    initial_state: InitialState,
    n_cycles: Sequence[int] | npt.NDArray,
    excited_qubits: Sequence[cirq.GridQubit],
    dt: float,
    h: float,
    mu: float,
    n_instances: int = 10,
    two_qubit_gate: TwoQubitGate = TwoQubitGate.CZ,
    basis: Basis = Basis.DUAL,
) -> List[cirq.Circuit]:
    """Generates the circuits needed for the 1D DFL experiment.

    Args:
        grid: The 1D sequence of qubits used in the experiment.
        initial_state: The initial state preparation. Use an Enum member from
            InitialState (SINGLE_SECTOR or SUPERPOSITION).
        n_cycles: The number of Trotter steps (cycles) to simulate.
        excited_qubits: Qubits to be excited in the initial state.
        dt: The time step size for the Trotterization.
        h: The gauge field strength coefficient.
        mu: The matter field strength coefficient.
        n_instances: The number of instances to generate.
        two_qubit_gate: The type of two-qubit gate to use in the Trotter step.
            Use an Enum member from TwoQubitGate (CZ or CPHASE).
        basis: The basis for the final circuit structure. Use an Enum member from
            Basis (LGT or DUAL).

    Returns:
        A list of all generated cirq.Circuit objects.

    Raises:
        ValueError: If an invalid option for `initial_state` or
            `basis` is given.
    """
    if initial_state.value == InitialState.SINGLE_SECTOR.value:
        initial_circuit = _energy_bump_initial_state(
            grid, h, InitialState.SINGLE_SECTOR, excited_qubits
        )
    elif initial_state.value == InitialState.SUPERPOSITION.value:
        initial_circuit = _energy_bump_initial_state(
            grid, h, InitialState.SUPERPOSITION, excited_qubits
        )
    else:
        raise ValueError("Invalid initial state")
    circuits = []
    for n_cycle in tqdm(n_cycles):
        print(int(np.max([0, n_cycle - 1])))
        circ = initial_circuit + trotter_circuit(
            grid, n_cycle, dt, h, mu, two_qubit_gate
        )
        if basis.value == Basis.LGT.value:
            circ += _change_basis(grid)
        elif basis.value == Basis.DUAL.value:
            pass
        else:
            raise ValueError("Invalid option for basis")
        for _ in range(n_instances):

            if basis.value == Basis.LGT.value:
                circ_z = circ + cirq.measure([q for q in grid], key="m")
            elif basis.value == Basis.DUAL.value:
                circ_z = (
                    circ
                    + _layer_hadamard(grid, "matter")
                    + cirq.measure([q for q in grid], key="m")
                )
            else:
                raise ValueError("Invalid option for basis")
            circ_x = (
                circ
                + _layer_hadamard(grid, "all")
                + cirq.measure([q for q in grid], key="m")
            )
            circuits.append(circ_z)
            circuits.append(circ_x)
    return circuits

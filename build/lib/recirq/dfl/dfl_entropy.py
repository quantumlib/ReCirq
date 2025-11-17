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

"""Run experiments for measuring second Renyi entropy via randomized measurements
in 1D DFL circuits.
"""

from typing import Sequence

import cirq
import numpy as np
import numpy.typing as npt
import os
import pickle

import cirq.transformers.randomized_measurements as rand_meas


def _layer_interaction(grid: Sequence[cirq.GridQubit],
                          dt: float,) -> cirq.Circuit:
    """Implements the ZZZ term of the DFL Hamiltonian
    Args:
        grid: The 1D sequence of qubits used in the experiment.
        dt: The time step size for the Trotterization. 

    Returns:
        cirq.Circuit for the Trotter evolution of the ZZZ term.
    """

    moment_1 = []
    moment_2 = []
    moment_3 = []
    moment_h = []
    for i in range(0, len(grid) // 2):
        q1 = grid[(2 * i)]
        q2 = grid[2 * i + 1]
        q3 = grid[(2 * i + 2) % n]
        moment_1.append(cirq.CZ(q1, q2))
        moment_2.append(cirq.CZ(q3, q2))
        moment_h.append(cirq.H(q2))
        moment_3.append(cirq.rx(2 * dt).on(q2))

    return cirq.Circuit.from_moments(
        cirq.Moment(moment_h),
        cirq.Moment(moment_1),
        cirq.Moment(moment_2),
        cirq.Moment(moment_3),
        cirq.Moment(moment_2),
        cirq.Moment(moment_1),
        cirq.Moment(moment_h))


def _layer_matter_gauge_x(
        grid: Sequence[cirq.GridQubit], dt: float, mu: float, h:float) -> cirq.Circuit:
    
    """Implements the matter and gauge fields

    Args:
        grid: The 1D sequence of qubits used in the experiment.
        dt: The time step size for the Trotterization.
        h: The gauge field strength coefficient.
        mu: The matter field strength coefficient.

    Returns:
        cirq.Circuit for the Trotter evolution of the matter and gauge fields.
    """

    moment = []
    for i in range(len(grid)):
        if i % 2 == 0:
            moment.append(cirq.rx(2 * mu*dt).on(grid[i]))
        else:
            moment.append(cirq.rx(2 * h*dt).on(grid[i]))
    return cirq.Circuit.from_moments(cirq.Moment(moment))


def _layer_floquet(grid: Sequence[cirq.GridQubit],
                   dt: float,
                   h: float,
                   mu: float) -> cirq.Circuit:
    """Constructs a Trotter circuit for 1D Disorder-Free Localization (DFL) simulation.

    Args:
        grid: The 1D sequence of qubits used in the experiment.
        dt: The time step size for the Trotterization.
        h: The gauge field strength coefficient.
        mu: The matter field strength coefficient.

    Returns:
        The complete cirq.Circuit for the Trotter evolution.

    """
    return _layer_interaction(grid,
                                 dt) + _layer_matter_gauge_x(grid, dt, mu, h)


def initial_state_for_entropy(grid: Sequence[cirq.GridQubit],
                              matter_config: str) -> cirq.Circuit:
    """Circuit for three types of initial states"""
    moment = []
    for i in range(len(grid)):
        if i % 2 == 0:
            if matter_config == "single_sector":
                moment.append(cirq.X(grid[i]))
                moment.append(cirq.H(grid[i]))

            elif matter_config == 'disordered':
                r = np.random.choice([0, 1])
                if r:
                    moment.append(cirq.X(grid[i]))
                moment.append(cirq.H(grid[i]))

            else:
                moment.append(cirq.I(grid[i]))
        else:
            if matter_config == "disordered":
                moment.append(cirq.X(grid[i]))
            moment.append(cirq.H(grid[i]))

    return cirq.Circuit(moment)


def get_1d_dfl_entropy_experiment_circuits(
        grid: Sequence[cirq.GridQubit],
        initial_state: str,
        n_cycle: int,
        dt: float,
        h: float,
        mu: float,
        n_basis: int = 100,
        gauge_compiling: bool = True) -> Sequence[cirq.Circuit]:

    if initial_state == "single_sector":
        initial_circuit = initial_state_for_entropy(grid, "single_sector")
    elif initial_state == "superposition":
        initial_circuit = initial_state_for_entropy(grid, "superposition")
    elif initial_state == "disordered":
        initial_circuit = initial_state_for_entropy(grid, "disordered")
    else:
        raise ValueError("Invalid initial state")

    circuits = []
    circ = initial_circuit + _layer_floquet(grid, dt, h, mu)*n_cycle

    for _ in range(n_basis):
        circ_randomized = rand_meas.RandomizedMeasurements()(
            circ, unitary_ensemble="Clifford")

        if gauge_compiling:
            circ_randomized = cirq.transformers.gauge_compiling.CZGaugeTransformer(
                circ_randomized
            )

        circ_merged = cirq.merge_single_qubit_moments_to_phxz(circ_randomized)
        circuits.append(circ_merged)

    return circuits


def run_1d_dfl_entropy_experiment_circuits(
        grid: Sequence[cirq.GridQubit],
        initial_states: Sequence[str],
        save_dir: str,
        n_cycles: Sequence[int] | npt.NDArray,
        dt: float,
        h: float,
        mu: float,
        n_basis: int = 100,
        n_shots: int = 1000,
        sampler: cirq.Sampler = cirq.Simulator(),
        gauge_compiling: bool = True) -> None:

    if not os.path.isdir(save_dir + "/dt{:.2f}".format(dt)):
        os.mkdir(save_dir + "/dt{:.2f}".format(dt))

    if not os.path.isdir(save_dir +
                         "/dt{:.2f}/h{:.2f}_mu{:.2f}".format(dt, h, mu)):
        os.mkdir(
            save_dir + "/dt{:.2f}/h{:.2f}_mu{:.2f}".format(dt, h, mu))

    for initial_state in initial_states:
        if not os.path.isdir(
            save_dir +
            "/dt{:.2f}/h{:.2f}_mu{:.2f}/{:s}".format(
                dt,
                h,
                mu,
                initial_state)):
            os.mkdir(
                save_dir +
                "/dt{:.2f}/h{:.2f}_mu{:.2f}/{:s}".format(
                    dt,
                    h,
                    mu,
                    initial_state))

        for n_cycle in n_cycles:
            print(initial_state, n_cycle)
            fname = (
                save_dir
                + "/dt{:.2f}/h{:.2f}_mu{:.2f}/{:s}/cycle{}.pickle".format(
                    dt, h, mu, initial_state, n_cycle
                )
            )
            circuits = get_1d_dfl_entropy_experiment_circuits(grid, initial_state=initial_state,
                                                              n_cycle=n_cycle, dt=dt, h=h, mu=mu,
                                                              n_basis=n_basis, gauge_compiling=gauge_compiling)

            results = sampler.run_batch(circuits, repetitions=n_shots)
            bitstrings = []
            for j in range(n_basis):
                bitstrings.append(
                    results[j][0].measurements["m"])

            with open(fname, "wb") as myfile:
                pickle.dump(bitstrings, myfile)
    return None

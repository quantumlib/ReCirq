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


from collections.abc import Sequence
from pathlib import Path
import pickle

import cirq
from cirq import add_dynamical_decoupling
from cirq.transformers import gauge_compiling, RandomizedMeasurements
import numpy as np
import numpy.typing as npt

from recirq.dfl.dfl_enums import InitialState


def _layer_interaction(
    grid: Sequence[cirq.GridQubit],
    dt: float,
) -> cirq.Circuit:
    """Implements the ZZZ term of the DFL Hamiltonian
    Each ZZZ term acts on matter-gauge-matter qubits.
    The resulting circuit for each term looks like:
        0: ───────@────────────────────────@───────
                  │                        │
        1: ───H───@───@───Rx(2 * dt)───@───@───H───
                      │                │
        2: ───────────@────────────────@───────────

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
        q3 = grid[(2 * i + 2) % len(grid)]
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
        cirq.Moment(moment_h),
    )


def _layer_matter_gauge_x(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    """Implements the X rotation for the matter and gauge qubits.
    The resulting circuit should look like:
        0: ──Rx(2*mu*dt)──

        1: ──Rx(2*h*dt)────

        2: ──Rx(2*mu*dt)───

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
            moment.append(cirq.rx(2 * mu * dt).on(grid[i]))
        else:
            moment.append(cirq.rx(2 * h * dt).on(grid[i]))
    return cirq.Circuit.from_moments(cirq.Moment(moment))


def layer_floquet(
    grid: Sequence[cirq.GridQubit], dt: float, h: float, mu: float
) -> cirq.Circuit:
    """Constructs a Trotter circuit for 1D Disorder-Free Localization (DFL) simulation.

    Args:
        grid: The 1D sequence of qubits used in the experiment.
        dt: Trotter step size.
        h: The coefficient on the gauge X term.
        mu: The coefficient on the matter sigma_x term.

    Returns:
        The complete cirq.Circuit for the Trotter evolution.

    """

    return _layer_interaction(grid, dt) + _layer_matter_gauge_x(grid, dt, h, mu)


def initial_state_for_entropy(
    grid: Sequence[cirq.GridQubit], matter_config: InitialState
) -> cirq.Circuit:
    """Circuit for three types of initial states:
    single_sector: |-> |+> |-> |+>...
    superposition: |0> |+> |0> |+>...
    disordered: |+/-> |-> |+/->... with randomly chosen |+> or |-> on matter sites
    """

    moment = []
    for i in range(len(grid)):
        if i % 2 == 0:
            if matter_config == InitialState.SINGLE_SECTOR:
                moment.append(cirq.X(grid[i]))
                moment.append(cirq.H(grid[i]))

            elif matter_config == InitialState.DISORDERED:
                r = np.random.choice([0, 1])
                if r:
                    moment.append(cirq.X(grid[i]))
                moment.append(cirq.H(grid[i]))

            else:
                moment.append(cirq.I(grid[i]))
        else:
            if matter_config == InitialState.DISORDERED:
                moment.append(cirq.X(grid[i]))
            moment.append(cirq.H(grid[i]))

    return cirq.Circuit(moment)


def get_1d_dfl_entropy_experiment_circuits(
    grid: Sequence[cirq.GridQubit],
    initial_state: InitialState,
    ncycles: int,
    dt: float,
    h: float,
    mu: float,
    n_basis: int = 100,
) -> Sequence[cirq.Circuit]:
    """Generate the circuit instances for the entropy experiment

    Args:
        grid: The qubits to use for the experiment.
        initial_state: Which initial state, see `InitialState` enum.
        ncycles: The number of Trotter steps (can be 0).
        dt: Trotter step size.
        h: The coefficient on the gauge X term.
        mu: The coefficient on the matter sigma_x term.
        n_basis: The number of random measurement bases to use.

    Returns:
        A list of the circuit instances.
    """

    initial_circuit = initial_state_for_entropy(grid, initial_state)
    circuits = []
    circ = initial_circuit + layer_floquet(grid, dt, h, mu) * ncycles

    for _ in range(n_basis):
        circ_randomized = RandomizedMeasurements()(circ, unitary_ensemble="Clifford")

        circuits.append(circ_randomized)

    return circuits


def run_1d_dfl_entropy_experiment(
    grid: Sequence[cirq.GridQubit],
    initial_states: Sequence[InitialState],
    save_dir: Path,
    n_cycles: Sequence[int] | npt.NDArray,
    dt: float,
    h: float,
    mu: float,
    n_basis: int = 100,
    n_shots: int = 1000,
    sampler: cirq.Sampler = cirq.Simulator(),
    gauge_compile: bool = True,
    dynamical_decouple: bool = True,
) -> None:
    """Run the 1D DFL experiment (Fig 4 of the paper).
    The paper is available at: https://arxiv.org/abs/2410.06557
    Saves the measurement bitstrings to save_dir.

    Data is saved in the following directory structure:
        save_dir/dt{dt}/h{h}_mu{mu}/{initial_state}/cycle{n_cycle}.pickle

    Attrs:
        grid: The qubits to use for the experiment.
        initial_states: The list of InitialState to use.
        save_dir: The directory in which to save the results.
        n_cycles: The list of number of Trotter steps to use.
        dt: The Trotter step size.
        h: The coefficient on the gauge X term.
        mu: The coefficient on the matter sigma_x term.
        n_basis: The number of random measurement bases to use.
        n_shots: The number of measurement shots to use.
        sampler: The cirq sampler to use.
        gauge_compile: Whether to apply gauge compiling.
        dynamical_decouple: Whether to apply dynamical decoupling.

    Returns:
        None
    """

    for initial_state in initial_states:
        dir_path = (
            save_dir / f"dt{dt:.2f}" / f"h{h:.2f}_mu{mu:.2f}" / initial_state.value
        )
        dir_path.mkdir(parents=True, exist_ok=True)

        for n_cycle in n_cycles:
            print("Initial state:", initial_state.value, "Cycle:", n_cycle)
            fname = dir_path / "cycle{}.pickle".format(n_cycle)
            circuits = get_1d_dfl_entropy_experiment_circuits(
                grid,
                initial_state=initial_state,
                ncycles=n_cycle,
                dt=dt,
                h=h,
                mu=mu,
                n_basis=n_basis,
            )

            circuits_modified = []
            for i in range(len(circuits)):
                circ_i = circuits[i]

                if gauge_compile:
                    circ_i = gauge_compiling.CZGaugeTransformer(circ_i)
                if dynamical_decouple:
                    circ_i = add_dynamical_decoupling(circ_i)
                circuits_modified.append(circ_i)

            results = sampler.run_batch(circuits, repetitions=n_shots)
            bitstrings = []
            for j in range(n_basis):
                bitstrings.append(results[j][0].measurements["m"])

            with open(fname, "wb") as myfile:
                pickle.dump(bitstrings, myfile)
    return None

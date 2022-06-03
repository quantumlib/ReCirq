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

from typing import Sequence, Tuple, List, Iterator

import functools
import numpy as np
import sympy as sp

import cirq
from recirq.time_crystals.dtcexperiment import DTCExperiment, comparison_experiments


def symbolic_dtc_circuit_list(
    qubits: Sequence[cirq.Qid], cycles: int
) -> List[cirq.Circuit]:
    """Create a list of symbolically parameterized dtc circuits, with increasing cycles

    Args:
        qubits: ordered sequence of available qubits, which are connected in a chain
        cycles: maximum number of cycles to generate up to

    Returns:
        list of circuits with `0, 1, 2, ... cycles` many cycles

    """
    num_qubits = len(qubits)

    # Symbol for g
    g_value = sp.Symbol("g")

    # Symbols for random variance (h) and initial state, one per qubit
    local_fields = sp.symbols(f"local_field_:{num_qubits}")
    initial_state = sp.symbols(f"initial_state_:{num_qubits}")

    # Symbols used for PhasedFsimGate, one for every qubit pair in the chain
    thetas = sp.symbols(f"theta_:{num_qubits - 1}")
    zetas = sp.symbols(f"zeta_:{num_qubits - 1}")
    chis = sp.symbols(f"chi_:{num_qubits - 1}")
    gammas = sp.symbols(f"gamma_:{num_qubits - 1}")
    phis = sp.symbols(f"phi_:{num_qubits - 1}")

    # Initial moment of Y gates, conditioned on initial state
    initial_operations = cirq.Moment(
        [cirq.Y(qubit) ** initial_state[index] for index, qubit in enumerate(qubits)]
    )

    # First component of U cycle, a moment of XZ gates.
    sequence_operations = []
    for index, qubit in enumerate(qubits):
        sequence_operations.append(
            cirq.PhasedXZGate(
                x_exponent=g_value,
                axis_phase_exponent=0.0,
                z_exponent=local_fields[index],
            )(qubit)
        )

    # Initialize U cycle
    u_cycle = [cirq.Moment(sequence_operations)]

    # Second and third components of U cycle, a chain of 2-qubit PhasedFSim gates
    #   The first component is all the 2-qubit PhasedFSim gates starting on even qubits
    #   The second component is the 2-qubit gates starting on odd qubits
    even_qubit_moment = []
    odd_qubit_moment = []
    for index, (qubit, next_qubit) in enumerate(zip(qubits, qubits[1:])):
        # Add an fsim gate
        coupling_gate = cirq.ops.PhasedFSimGate(
            theta=thetas[index],
            zeta=zetas[index],
            chi=chis[index],
            gamma=gammas[index],
            phi=phis[index],
        )

        if index % 2:
            even_qubit_moment.append(coupling_gate.on(qubit, next_qubit))
        else:
            odd_qubit_moment.append(coupling_gate.on(qubit, next_qubit))

    # Add the two components into the U cycle
    u_cycle.append(cirq.Moment(even_qubit_moment))
    u_cycle.append(cirq.Moment(odd_qubit_moment))

    # Prepare a list of circuits, with n=0,1,2,3 ... cycles many cycles
    circuit_list = []
    total_circuit = cirq.Circuit(initial_operations)
    circuit_list.append(total_circuit.copy())
    for _ in range(cycles):
        for moment in u_cycle:
            total_circuit.append(moment)
        circuit_list.append(total_circuit.copy())

    return circuit_list


def simulate_dtc_circuit_list(
    circuit_list: Sequence[cirq.Circuit],
    param_resolver: cirq.ParamResolver,
    qubit_order: Sequence[cirq.Qid],
    simulator: "cirq.SimulatesIntermediateState" = None,
) -> np.ndarray:
    """Simulate a dtc circuit list for a particular param_resolver

        Utilizes the fact that simulating the last circuit in the list also
            simulates each previous circuit along the way

    Args:
        circuit_list: DTC circuit list; each element is a circuit with
            increasingly many cycles
        param_resolver: `cirq.ParamResolver` to resolve symbolic parameters
        qubit_order: ordered sequence of qubits connected in a chain
        simulator: Optional simulator object which must
            support the `simulate_moment_steps` function

    Returns:
        `np.ndarray` of shape (len(circuit_list), 2**number of qubits) representing
            the probability of measuring each bit string, for each circuit in the list

    """
    # prepare simulator
    if simulator is None:
        simulator = cirq.Simulator()

    # record lengths of circuits in list
    if not all(len(x) < len(y) for x, y in zip(circuit_list, circuit_list[1:])):
        raise ValueError("circuits in circuit_list are not in increasing order of size")
    circuit_positions = {len(c) - 1 for c in circuit_list}

    # only simulate one circuit, the last one
    circuit = circuit_list[-1]

    # use simulate_moment_steps to recover all of the state vectors necessary,
    #   while only simulating the circuit list once
    probabilities = []
    for k, step in enumerate(
        simulator.simulate_moment_steps(
            circuit=circuit, param_resolver=param_resolver, qubit_order=qubit_order
        )
    ):
        # add the state vector if the number of moments simulated so far is equal
        #   to the length of a circuit in the circuit_list
        if k in circuit_positions:
            probabilities.append(np.abs(step.state_vector()) ** 2)

    return np.asarray(probabilities)


def simulate_dtc_circuit_list_sweep(
    circuit_list: Sequence[cirq.Circuit],
    param_resolvers: Sequence[cirq.ParamResolver],
    qubit_order: Sequence[cirq.Qid],
) -> Iterator[np.ndarray]:
    """Simulate a dtc circuit list over a sweep of param_resolvers

    Args:
        circuit_list: DTC circuit list; each element is a circuit with
            increasingly many cycles
        param_resolvers: list of `cirq.ParamResolver`s to sweep over
        qubit_order: ordered sequence of qubits connected in a chain

    Yields:
        for each param_resolver, `np.ndarray`s of shape
            (len(circuit_list), 2**number of qubits) representing the probability
            of measuring each bit string, for each circuit in the list

    """
    # iterate over param resolvers and simulate for each
    for param_resolver in param_resolvers:
        yield simulate_dtc_circuit_list(circuit_list, param_resolver, qubit_order)


def get_polarizations(
    probabilities: np.ndarray,
    num_qubits: int,
    initial_states: np.ndarray = None,
) -> np.ndarray:
    """Get polarizations from matrix of probabilities, possibly autocorrelated on
        the initial state.

    A polarization is the marginal probability for a qubit to measure zero or one,
        over all possible basis states, scaled to the range [-1. 1].

    Args:
        probabilities: `np.ndarray` of shape (:, cycles, 2**qubits)
            representing probability to measure each bit string
        num_qubits: the number of qubits in the circuit the probabilities
            were generated from
        initial_states: `np.ndarray` of shape (:, qubits) representing the initial
            state for each dtc circuit list

    Returns:
        `np.ndarray` of shape (:, cycles, qubits) that represents each
            qubit's polarization

    """
    # prepare list of polarizations for each qubit
    polarizations = []
    for qubit_index in range(num_qubits):
        # select all indices in range(2**num_qubits) for which the
        #   associated element of the statevector has qubit_index as zero
        shift_by = num_qubits - qubit_index - 1
        state_vector_indices = [
            i for i in range(2**num_qubits) if not (i >> shift_by) % 2
        ]

        # sum over all probabilities for qubit states for which qubit_index is zero,
        #   and rescale them to [-1,1]
        polarization = (
            2.0
            * np.sum(
                probabilities.take(indices=state_vector_indices, axis=-1),
                axis=-1,
            )
            - 1.0
        )
        polarizations.append(polarization)

    # turn polarizations list into an array,
    #   and move the new, leftmost axis for qubits to the end
    polarizations = np.moveaxis(np.asarray(polarizations), 0, -1)

    # flip polarizations according to the associated initial_state, if provided
    #   this means that the polarization of a qubit is relative to it's initial state
    if initial_states is not None:
        initial_states = 1 - 2.0 * initial_states
        polarizations = initial_states * polarizations

    return polarizations


def signal_ratio(zeta_1: np.ndarray, zeta_2: np.ndarray) -> np.ndarray:
    """Calculate signal ratio between two signals

    Signal ratio measures how different two signals are,
        proportional to how large they are.

    Args:
        zeta_1: signal (`np.ndarray` to represent polarization over time)
        zeta 2: signal (`np.ndarray` to represent polarization over time)

    Returns:
        computed ratio of the signals zeta_1 and zeta_2 (`np.ndarray`)
            to represent polarization over time)

    """
    return np.abs(zeta_1 - zeta_2) / (np.abs(zeta_1) + np.abs(zeta_2))


def simulate_for_polarizations(
    dtcexperiment: DTCExperiment,
    circuit_list: Sequence[cirq.Circuit],
    autocorrelate: bool = True,
    take_abs: bool = False,
) -> np.ndarray:
    """Simulate and get polarizations for a single DTCExperiment and circuit list

    Args:
        dtcexperiment: DTCExperiment noting the parameters to simulate over some
            number of disorder instances
        circuit_list: symbolic dtc circuit list
        autocorrelate: whether or not to autocorrelate the polarizations with their
            respective initial states
        take_abs: whether or not to take the absolute value of the polarizations

    Returns:
        simulated polarizations (np.ndarray of shape (num_cycles, num_qubits)) from
            the experiment, averaged over disorder instances

    """
    # create param resolver sweep
    param_resolvers = dtcexperiment.param_resolvers()

    # prepare simulation generator
    probabilities_generator = simulate_dtc_circuit_list_sweep(
        circuit_list, param_resolvers, dtcexperiment.qubits
    )

    # map get_polarizations over probabilities_generator
    polarizations_generator = map(
        lambda probabilities, initial_state: get_polarizations(
            probabilities,
            num_qubits=len(dtcexperiment.qubits),
            initial_states=(initial_state if autocorrelate else None),
        ),
        probabilities_generator,
        dtcexperiment.initial_states,
    )

    # take sum of (absolute value of) polarizations over different disorder instances
    polarization_sum = functools.reduce(
        lambda x, y: x + (np.abs(y) if take_abs else y),
        polarizations_generator,
        np.zeros((len(circuit_list), len(dtcexperiment.qubits))),
    )

    # get average over disorder instances
    disorder_averaged_polarizations = (
        polarization_sum / dtcexperiment.disorder_instances
    )

    return disorder_averaged_polarizations


def run_comparison_experiment(
    qubits: Sequence[cirq.Qid],
    cycles: int,
    disorder_instances: int,
    autocorrelate: bool = True,
    take_abs: bool = False,
    **kwargs,
) -> Iterator[np.ndarray]:
    """Run multiple DTC experiments for qubit polarizations over different parameters.

    This uses the default parameter options noted in
        `dtcexperiment.comparison_experiments` for any parameter not supplied in
        kwargs. A DTC experiment is then created and simulated for each possible
        parameter combination before qubit polarizations by DTC cycle are
        computed and yielded. Each yield is an `np.ndarray` of shape (qubits, cycles)
        for a specific combination of parameters.

    Args:
        qubits: ordered sequence of available qubits, which are connected in a chain
        cycles: maximum number of cycles to generate up to
        autocorrelate: whether or not to autocorrelate the polarizations with their
            respective initial states
        take_abs: whether or not to take the absolute value of the polarizations
        kwargs: lists of non-default argument configurations to pass through
            to `dtcexperiment.comparison_experiments`

    Yields:
        disorder averaged polarizations, ordered by
            `dtcexperiment.comparison_experiments`, with all other parameters default

    """
    circuit_list = symbolic_dtc_circuit_list(qubits, cycles)
    for dtcexperiment in comparison_experiments(
        qubits=qubits, disorder_instances=disorder_instances, **kwargs
    ):
        yield simulate_for_polarizations(
            dtcexperiment=dtcexperiment,
            circuit_list=circuit_list,
            autocorrelate=autocorrelate,
            take_abs=take_abs,
        )

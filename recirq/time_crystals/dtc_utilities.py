# Copyright 2021 Google
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

from typing import Sequence, Tuple, List

import cirq
import functools
from recirq.time_crystals.dtctask import DTCTask, CompareDTCTask
import numpy as np
import sympy as sp

def symbolic_dtc_circuit_list(
        qubits: Sequence[cirq.Qid],
        cycles: int
        ) -> List[cirq.Circuit]:

    """ Create a list of symbolically parameterized dtc circuits, with increasing cycles
    Args:
        qubits: ordered sequence of available qubits, which are connected in a chain
        cycles: maximum number of cycles to generate up to
    Returns:
        list of circuits with `0, 1, 2, ... cycles` many cycles
    """

    num_qubits = len(qubits)

    # Symbol for g
    g_value = sp.Symbol('g')

    # Symbols for random variance (h) and initial state, one per qubit
    local_fields = sp.symbols('local_field_:' + str(num_qubits))
    initial_state = sp.symbols('initial_state_:' + str(num_qubits))

    # Symbols used for PhasedFsimGate, one for every qubit pair in the chain
    thetas = sp.symbols('theta_:' + str(num_qubits - 1))
    zetas = sp.symbols('zeta_:' + str(num_qubits - 1))
    chis = sp.symbols('chi_:' + str(num_qubits - 1))
    gammas = sp.symbols('gamma_:' + str(num_qubits - 1))
    phis = sp.symbols('phi_:' + str(num_qubits - 1))

    # Initial moment of Y gates, conditioned on initial state
    initial_operations = cirq.Moment([cirq.Y(qubit) ** initial_state[index] for index, qubit in enumerate(qubits)])

    # First component of U cycle, a moment of XZ gates.
    sequence_operations = []
    for index, qubit in enumerate(qubits):
        sequence_operations.append(cirq.PhasedXZGate(
                x_exponent=g_value, axis_phase_exponent=0.0,
                z_exponent=local_fields[index])(qubit))

    # Initialize U cycle
    u_cycle = [cirq.Moment(sequence_operations)]

    # Second and third components of U cycle, a chain of 2-qubit PhasedFSim gates
    #   The first component is all the 2-qubit PhasedFSim gates starting on even qubits
    #   The second component is the 2-qubit gates starting on odd qubits
    operation_list, other_operation_list = [],[]
    previous_qubit, previous_index = None, None
    for index, qubit in enumerate(qubits):
        if previous_qubit is None:
            previous_qubit, previous_index = qubit, index
            continue

        # Add an fsim gate
        coupling_gate = cirq.ops.PhasedFSimGate(
            theta=thetas[previous_index],
            zeta=zetas[previous_index],
            chi=chis[previous_index],
            gamma=gammas[previous_index],
            phi=phis[previous_index]
        )
        operation_list.append(coupling_gate.on(previous_qubit, qubit))

        # Swap the operation lists, to avoid two-qubit gate overlap
        previous_qubit, previous_index = qubit, index
        operation_list, other_operation_list = other_operation_list, operation_list

    # Add the two components into the U cycle
    u_cycle.append(cirq.Moment(operation_list))
    u_cycle.append(cirq.Moment(other_operation_list))

    # Prepare a list of circuits, with n=0,1,2,3 ... cycles many cycles
    circuit_list = []
    total_circuit = cirq.Circuit(initial_operations)
    circuit_list.append(total_circuit.copy())
    for c in range(cycles):
        for m in u_cycle:
            total_circuit.append(m)
        circuit_list.append(total_circuit.copy())

    return circuit_list

def simulate_dtc_circuit_list(circuit_list: Sequence[cirq.Circuit], param_resolver: cirq.ParamResolver, qubit_order: Sequence[cirq.Qid]) -> np.ndarray:
    """ Simulate a dtc circuit list for a particular param_resolver
        Depends on the fact that simulating the last circuit in the list also simulates each previous circuit along the way
    Args:
        circuit_list: DTC circuit list; each element is a circuit with increasingly many cycles
        param_resolver: `cirq.ParamResolver` to resolve symbolic parameters
        qubit_order: ordered sequence of qubits connected in a chain
    Returns:
        `np.ndarray` of shape (len(circuit_list), 2**number of qubits) representing the probability of measuring each bit string, for each circuit in the list
    """

    # prepare simulator
    simulator = cirq.Simulator()

    # record lengths of circuits in list
    circuit_positions = [len(c) - 1 for c in circuit_list]

    # only simulate one circuit, the last one
    circuit = circuit_list[-1]

    # use simulate_moment_steps to recover all of the state vectors necessary, while only simulating the circuit list once
    probabilities = []
    for k, step in enumerate(simulator.simulate_moment_steps(circuit=circuit, param_resolver=param_resolver, qubit_order=qubit_order)):
        # add the state vector if the number of moments simulated so far is equal to the length of a circuit in the circuit_list
        if k in circuit_positions:
            probabilities.append(np.abs(step.state_vector()) ** 2)

    return np.asarray(probabilities)

def simulate_dtc_circuit_list_sweep(circuit_list: Sequence[cirq.Circuit], param_resolvers: Sequence[cirq.ParamResolver], qubit_order: Sequence[cirq.Qid]):
    """ Simulate a dtc circuit list over a sweep of param_resolvers
    Args:
        circuit_list: DTC circuit list; each element is a circuit with increasingly many cycles
        param_resolvers: list of `cirq.ParamResolver`s to sweep over
        qubit_order: ordered sequence of qubits connected in a chain
    Yields:
        for each param_resolver, `np.ndarray`s of shape (len(circuit_list), 2**number of qubits) representing the probability of measuring each bit string, for each circuit in the list
    """

    # iterate over param resolvers and simulate for each
    for param_resolver in param_resolvers:
      yield simulate_dtc_circuit_list(circuit_list, param_resolver, qubit_order)

def get_polarizations(probabilities: np.ndarray, num_qubits: int, cycles_axis: int = -2, probabilities_axis: int = -1, initial_states: np.ndarray = None) -> np.ndarray:
    """ Get polarizations from matrix of probabilities, possibly autocorrelated on the initial state
    Args:
        probabilities: `np.ndarray` of shape (:, cycles, probabilities) representing probability to measure each bit string
        num_qubits: the number of qubits in the circuit the probabilities were generated from
        cycles_axis: the axis that represents the dtc cycles (if not in -2 indexed axis)
        probabilities_axis: the axis that represents the probabilities for each bit string (if not in -1 indexed axis)
        initial_states: `np.ndarray` of shape (:, qubits) representing the initial state for each dtc circuit list
    Returns:
        `np.ndarray` of shape (:, cycles, qubits) that represents each qubit's polarization
    """

    # prepare list of polarizations for each qubit
    polarizations = []
    for qubit_index in range(num_qubits):
        # select all indices in range(2**num_qubits) for which the associated element of the statevector has qubit_index as zero
        shift_by = num_qubits - qubit_index - 1
        state_vector_indices = [i for i in range(2 ** num_qubits) if not (i >> shift_by) % 2]

        # sum over all amplitudes for qubit states for which qubit_index is zero, and rescale them to [-1,1]
        polarization = 2.0 * np.sum(probabilities.take(indices=state_vector_indices, axis=probabilities_axis), axis=probabilities_axis) - 1.0
        polarizations.append(polarization)

    # turn polarizations list into an array, and move the new, leftmost axis for qubits to probabilities_axis
    polarizations = np.moveaxis(np.asarray(polarizations), 0, probabilities_axis)

    # flip polarizations according to the associated initial_state, if provided
    #   this means that the polarization of a qubit is relative to it's initial state
    if initial_states is not None:
        initial_states = 1 - 2.0 * initial_states
        polarizations = initial_states * polarizations

    return polarizations


def signal_ratio(zeta_1: np.ndarray, zeta_2: np.ndarray):
    ''' Calculate signal ratio between two signals
    Args:
        zeta_1: signal (`np.ndarray` to represent polarization over time)
        zeta 2: signal (`np.ndarray` to represent polarization over time)
    Returns:
        computed ratio signal of zeta_1 and zeta_2 (`np.ndarray` to represent polarization over time)
    '''

    return np.abs(zeta_1 - zeta_2)/(np.abs(zeta_1) + np.abs(zeta_2))


def simulate_for_polarizations(dtctask: DTCTask, circuit_list: Sequence[cirq.Circuit], autocorrelate: bool = True, take_abs: bool = False):
    """ Simulate and get polarizations for a single DTCTask and circuit list
    Args:
        dtctask: DTCTask noting the parameters to simulate over some number of disorder instances
        circuit_list: symbolic dtc circuit list
        autocorrelate: whether or not to autocorrelate the polarizations with their respective initial states
        take_abs: whether or not to take the absolute value of the polarizations
    Returns:
        simulated polarizations (np.ndarray of shape (num_cycles, num_qubits)) from the experiment, averaged over disorder instances
    """

    # create param resolver sweep
    param_resolvers = dtctask.param_resolvers()

    # prepare simulation generator
    probabilities_generator = simulate_dtc_circuit_list_sweep(circuit_list, param_resolvers, dtctask.qubits)

    # map get_polarizations over probabilities_generator
    polarizations_generator = map(lambda probabilities, initial_state:
                                get_polarizations(probabilities, num_qubits=len(dtctask.qubits), cycles_axis=0, probabilities_axis=1, initial_states=(initial_state if autocorrelate else None)),
                                 probabilities_generator, dtctask.initial_states)

    # take sum of (absolute value of) polarizations over different disorder instances
    polarization_sum = functools.reduce(lambda x,y: x+(np.abs(y) if take_abs else y), polarizations_generator, np.zeros((len(circuit_list), len(dtctask.qubits))))

    # get average over disorder instances
    disorder_averaged_polarizations = polarization_sum / dtctask.disorder_instances

    return disorder_averaged_polarizations


def run_comparison_experiment(comparedtctask: CompareDTCTask, autocorrelate: bool = True, take_abs: bool = False):
    """ Run comparison experiment from a CompareDTCTask
    Args:
        comparedtctask: CompareDTCTask which notes which dtc arguments to compare, and default arguments
        autocorrelate: whether or not to autocorrelate the polarizations with their respective initial states
        take_abs: whether or not to take the absolute value of the polarizations
    Yields:
        disorder averaged polarizations, in order of the product of options supplied to comparedtctask, with all other parameters default
    """

    for dtctask in comparedtctask.dtctasks():
        yield simulate_for_polarizations(dtctask=dtctask, circuit_list=comparedtctask.circuit_list, autocorrelate=autocorrelate, take_abs=take_abs)

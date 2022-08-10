# Copyright 2020 Google
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

"""Functions for performing parallel cross entropy benchmarking."""
from dataclasses import dataclass
from typing import Sequence, List, Set, Tuple, Dict, Union, Optional

import cirq
import numpy as np
import pybobyqa
from matplotlib import pyplot as plt

from recirq.otoc.utils import (
    bits_to_probabilities,
    angles_to_fsim,
    pauli_error_fit,
    generic_fsim_gate,
    cz_to_sqrt_iswap,
)

_rot_ops = [
    cirq.X ** 0.5,
    cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
    cirq.Y ** 0.5,
    cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.5),
    cirq.X ** -0.5,
    cirq.PhasedXPowGate(phase_exponent=-0.75, exponent=0.5),
    cirq.Y ** -0.5,
    cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=0.5),
]
_rot_mats = [cirq.unitary(r) for r in _rot_ops]
_fsim_angle_labels = [
    "theta",
    "delta_plus",
    "delta_minus_off_diag",
    "delta_minus_diag",
    "phi",
]


@dataclass
class XEBData:
    """Class for storing the cycle-dependent fidelities and purities of an XEB experiment.

    Each field contains two 1D arrays representing the circuit length (i.e. number of cycles) and
    the corresponding fidelity or purity values. Fields containing 'fit' in their names are
    fitting results to an exponential decay.
    """

    fidelity_optimized: Tuple[np.ndarray, np.ndarray]
    fidelity_optimized_fit: Tuple[np.ndarray, np.ndarray]
    fidelity_unoptimized: Tuple[np.ndarray, np.ndarray]
    fidelity_unoptimized_fit: Tuple[np.ndarray, np.ndarray]
    purity: Tuple[np.ndarray, np.ndarray]
    purity_fit: Tuple[np.ndarray, np.ndarray]


@dataclass
class ParallelXEBResults:
    """Class for storing results of a parallel-XEB experiment."""

    fitted_gates: Dict[Tuple[Tuple[int, int], Tuple[int, int]], cirq.Circuit]
    correction_gates: Dict[Tuple[Tuple[int, int], Tuple[int, int]], cirq.Circuit]
    fitted_angles: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]]
    final_errors_optimized: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]
    final_errors_unoptimized: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]
    purity_errors: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]
    raw_data: Dict[Tuple[Tuple[int, int], Tuple[int, int]], XEBData]


def plot_xeb_results(xeb_results: ParallelXEBResults) -> None:
    """Plots the results of a parallel XEB experiment."""
    for (q0, q1), xeb_data in xeb_results.raw_data.items():

        # Plot the fidelities (both with unoptimized and optimized two-qubit unitaries) and speckle
        # purities as functions of XEB cycles, for each qubit pair.
        err_0 = xeb_results.final_errors_unoptimized[(q0, q1)]
        err_1 = xeb_results.final_errors_optimized[(q0, q1)]
        err_p = xeb_results.purity_errors[(q0, q1)]

        fig = plt.figure()
        plt.plot(
            xeb_data.fidelity_unoptimized[0],
            xeb_data.fidelity_unoptimized[1],
            "ro",
            figure=fig,
            label=r"{} and {}, unoptimized [$r_p$ = {}]".format(q0, q1, err_0.__round__(5)),
        )
        plt.plot(
            xeb_data.fidelity_optimized[0],
            xeb_data.fidelity_optimized[1],
            "bo",
            figure=fig,
            label=r"{} and {}, optimized [$r_p$ = {}]".format(q0, q1, err_1.__round__(5)),
        )
        plt.plot(
            xeb_data.purity[0],
            xeb_data.purity[1],
            "go",
            figure=fig,
            label=r"{} and {}, purity error = {}".format(q0, q1, err_p.__round__(5)),
        )
        plt.plot(xeb_data.fidelity_unoptimized_fit[0], xeb_data.fidelity_unoptimized_fit[1], "r--")
        plt.plot(xeb_data.fidelity_optimized_fit[0], xeb_data.fidelity_optimized_fit[1], "b--")
        plt.plot(xeb_data.purity_fit[0], xeb_data.purity_fit[1], "g--")
        plt.legend()
        plt.xlabel("Number of Cycles")
        plt.ylabel(r"XEB Fidelity")

    num_pairs = len(list(xeb_results.final_errors_optimized.keys()))
    pair_pos = np.linspace(0, 1, num_pairs)

    # Plot the integrated histogram of Pauli errors for all pairs.
    fig_0 = plt.figure()
    plt.plot(
        sorted(xeb_results.final_errors_unoptimized.values()),
        pair_pos,
        figure=fig_0,
        label="Unoptimized Unitaries",
    )
    plt.plot(
        sorted(xeb_results.final_errors_optimized.values()),
        pair_pos,
        figure=fig_0,
        label="Optimized Unitaries",
    )
    plt.plot(
        sorted(xeb_results.purity_errors.values()),
        pair_pos,
        figure=fig_0,
        label="Purity Errors",
    )
    plt.xlabel(r"Pauli Error Rate, $r_p$")
    plt.ylabel(r"Integrated Histogram")
    plt.legend()

    # Plot the shifts in the FSIM angles derived from fitting the XEB data.
    fig_1 = plt.figure()
    for label in _fsim_angle_labels:
        shifts = [a[label] for a in xeb_results.fitted_angles.values()]
        plt.plot(sorted(shifts), pair_pos, figure=fig_1, label=label)
    plt.xlabel(r"FSIM Angle Error (Radian)")
    plt.ylabel(r"Integrated Histogram")
    plt.legend()


def build_xeb_circuits(
    qubits: Sequence[cirq.GridQubit],
    cycles: Sequence[int],
    benchmark_ops: Sequence[Union[cirq.Moment, Sequence[cirq.Moment]]] = None,
    random_seed: int = None,
    sq_rand_nums: Optional[np.ndarray] = None,
    reverse: bool = False,
    z_only: bool = False,
    ancilla: Optional[cirq.GridQubit] = None,
    cycles_per_echo: Optional[int] = None,
    light_cones: Optional[List[List[Set[cirq.GridQubit]]]] = None,
    echo_indices: Optional[np.ndarray] = None,
) -> Tuple[List[cirq.Circuit], np.ndarray]:
    r"""Builds random circuits for cross entropy benchmarking (XEB).

    A list of cirq.Circuits of varying lengths are generated, which are made of random
    single-qubit gates and optional two-qubit gates.

    Args:
        qubits: The qubits to be involved in XEB.
        cycles: The different numbers of cycles the random circuits will have.
        benchmark_ops: The operations to be inserted between random single-qubit gates. They can
            be one or more cirq.Moment objects, or None (in which case no operation will be
            inserted between the random single-qubit gates).
        random_seed: The random seed for the single-qubit gates. If unspecified, no random seed
            will be used.
        sq_rand_nums: The random numbers representing the single-qubit gates. They must be
            integers from 0 to 7 if the z_only is False, and floats between -1 and 1 if z_only is
            True. The dimension of sq_rand_nums should be len(qubits) by max(cycles). If
            unspecified, the gates will be generated in-situ at random.
        reverse: If True, benchmark_ops will be applied before the random single-qubit gates in
            each cycle. Otherwise, it will be applied after the random single-qubit gates.
        z_only: Whether the single-qubit gates are to be random \pi/2 rotations around axes on
            the equatorial plane of the Bloch sphere (z_only = False), or random rotations around
            the z-axis (z_only = True). In the former case, the axes of rotations will be chosen
            randomly from 8 evenly spaced axes ($\pi/4$, $\pi/2$ ... $7\pi/4$ radians from the
            x-axis). In the latter case, the angles of rotation will be any random value between
            $-\pi$ and $\pi$.
        ancilla: If specified, an additional qubit will be included in the circuit which does not
            interact with the other qubits and only has spin-echo pulses applied to itself.
        cycles_per_echo: How often a spin-echo (Y gate) gate is to be applied to the ancilla
            qubit. For example, if the value is 2, a Y gate will be applied every other cycle.
        light_cones: A list of length 1 or 2, each specifying a lightcone correponding to a list
            of sets of qubits with the same length as max(cycles). For each cycle, single-qubit
            gates outside the first lightcone are either removed or replaced with a spin-echo
            pulse. Single-qubit gates outside the second lightcone, if specified, are always
            removed.
        echo_indices: An array with the same dimension as sq_rand_nums and random integer values
            of 1, 2, 3 or 4. They specify the spin-echo pulses applied to qubits outside the
            first lightcone, which can be +/-X or +/-Y gates.

    Returns:
        all_circuits: A list of random circuits, each containing a specified number of cycles.
        sq_gate_indices: An NxM array, where N is the number of qubits and M is the maximnum
            number of cycles. The array elements are the indices for the random single-qubit gates.
    """
    if light_cones is not None:
        if len(light_cones) > 2:
            raise ValueError("light_cones may only have length 1 or 2")

    if benchmark_ops is not None:
        num_d = len(benchmark_ops)
    else:
        num_d = 0
    max_cycles = max(cycles)

    single_rots, sq_gate_indices = _random_rotations(
        qubits,
        max_cycles,
        random_seed,
        sq_rand_nums,
        light_cones,
        echo_indices,
        z_rotations_only=z_only,
    )

    all_circuits = []  # type: List[cirq.Circuit]
    for num_cycles in cycles:
        circuit_exp = cirq.Circuit()
        for i in range(num_cycles):
            c = i + 1 if not reverse else num_cycles - i
            if ancilla is not None and cycles_per_echo is not None:
                if c % cycles_per_echo == 0:
                    op_list = [cirq.Y(ancilla)]
                    op_list.extend(single_rots[i])
                    rots = cirq.Moment(op_list)
                else:
                    rots = cirq.Moment(single_rots[i])
            else:
                rots = cirq.Moment(single_rots[i])
            if reverse:
                if benchmark_ops is not None:
                    circuit_exp.append(benchmark_ops[i % num_d])
                circuit_exp.append(rots, strategy=cirq.InsertStrategy.NEW)
            else:
                circuit_exp.append(rots, strategy=cirq.InsertStrategy.NEW)
                if benchmark_ops is not None:
                    circuit_exp.append(benchmark_ops[i % num_d])
        all_circuits.append(circuit_exp)
    return all_circuits, sq_gate_indices


def parallel_xeb_fidelities(
    all_qubits: List[Tuple[int, int]],
    num_cycle_range: Sequence[int],
    measured_bits: List[List[List[np.ndarray]]],
    scrambling_gates: List[List[np.ndarray]],
    fsim_angles: Dict[str, float],
    interaction_sequence: Optional[
        List[Set[Tuple[Tuple[float, float], Tuple[float, float]]]]
    ] = None,
    gate_to_fit: str = "iswap",
    num_restarts: int = 3,
    num_points: int = 8,
    print_fitting_progress: bool = True,
) -> ParallelXEBResults:
    """Computes and optimizes cycle fidelities from parallel XEB data.

    Args:
        all_qubits: List of qubits involved in a parallel XEB experiment, specified using their
            (row, col) locations.
        num_cycle_range: The different numbers of cycles in the random circuits.
        measured_bits: The experimental bit-strings stored in a nested list. The first dimension
            of the nested list represents different configurations (e.g. how the two-qubit gates
            are applied) used in parallel XEB. The second dimension represents different trials
            (i.e. random circuit instances) used in XEB. The third dimension represents the
            different numbers of cycles and must be the same as len(num_cycle_range). Each
            np.ndarray has dimension M x N, where M is the number of repetitions (stats) for each
            circuit and N is the number of qubits involved.
        scrambling_gates: The random circuit indices specified as integers between 0 and 7. See
            the documentation of build_xeb_circuits for details. The first dimension of the
            nested list represents the different configurations and must be the same as the first
            dimension of measured_bits. The second dimension represents the different trials and
            must be the same as the second dimension of measured_bits.
        fsim_angles: An initial guess for the five FSIM angles for each qubit pair.
        interaction_sequence: The pairs of qubits with FSIM applied for each configuration. Must
            be the same as len(measured_bits).
        gate_to_fit: Can be either 'iswap', 'sqrt-iswap', 'cz' or any other string. Determines
            the FSIM angles that will be changed from their initial guess values to optimize the
            XEB fidelity of each qubit pair. For 'iswap', only 'delta_plus' and
            'delta_minus_off_diag' are changed. For 'sqrt-iswap', 'delta_plus',
            'delta_minus_off_diag' and 'delta_minus_diag' are changed. For 'cz',
            only 'delta_plus' and 'delta_minus_diag' are changed. For any other string, all five
            angles are changed.
        num_restarts: Number of restarts with different random initial guesses.
        num_points: The total number of XEB fidelities to be used in the cost function for
            optimization. Default is 8, such that the cost function is the sum of the XEB
            fidelities for the first 8 numbers of cycles in num_cycle_range.
        print_fitting_progress: Whether to print progress during the fitting process.

    Returns:
        A ParallelXEBResults object that contains the following fields:
        fitted_gates: A dictionary with qubit pairs as keys and optimized FSIM unitaries,
            represented by cirq.Circuit objects, as values.
        correction_gates: Same as fitted_gates, but with all Z rotations reversed in signs.
        fitted_angles: A dictionary with qubit pairs as keys and optimized FSIM unitaries as
            values. Here the FSIM unitaries are represented as a dictionaries with the names of
            the FSIM phases as keys and their fitted values as values.
        final_errors_optimized: A dictionary with qubit pairs as keys and their cycle errors
            after fitting as values.
        final_errors_unoptimized: A dictionary with qubit pairs as keys and their cycle errors
            before fitting as values.
        purity_errors: A dictionary with qubit pairs as keys and their speckle purity errors per
            cycle as values.
        raw_data: A dictionary with qubit pairs as keys and XEBData as values. Each XEBData
            contains the cycle-dependent XEB fidelities and purities, as well as their fits.
    """
    num_trials = len(measured_bits[0])
    p_data_all, sq_gates = _pairwise_xeb_probabilities(
        all_qubits,
        num_cycle_range,
        measured_bits,
        scrambling_gates,
        interaction_sequence,
    )
    final_errors_unoptimized = {}
    final_errors_optimized = {}
    delta_angles = {}
    purity_errors = {}
    fitted_gates = {}
    fitted_angles = {}
    correction_gates = {}
    raw_data = {}

    for (q0, q1), p_data in p_data_all.items():
        if print_fitting_progress:
            print("Fitting qubits {} and {}".format(q0, q1))

        def xeb_fidelity(
            angle_shifts: np.ndarray, num_p: int
        ) -> Tuple[float, List[float], np.ndarray, np.ndarray, float]:
            new_angles = fsim_angles.copy()
            for i, angle_name in enumerate(_fsim_angle_labels):
                new_angles[angle_name] += angle_shifts[i]
            fsim_mat = angles_to_fsim(**new_angles)

            max_cycles = num_cycle_range[num_p - 1]

            p_sim = [np.zeros((num_trials, 4)) for _ in range(max_cycles)]
            for i in range(num_trials):
                unitary = np.identity(4, dtype=complex)
                for j in range(max_cycles):
                    mat_0 = _rot_mats[sq_gates[(q0, q1)][i][0, j]]
                    mat_1 = _rot_mats[sq_gates[(q0, q1)][i][1, j]]
                    unitary = np.kron(mat_0, mat_1).dot(unitary)
                    unitary = fsim_mat.dot(unitary)
                    if j + 1 in num_cycle_range:
                        idx = num_cycle_range.index(j + 1)
                        p_sim[idx][i, :] = np.abs(unitary[:, 0]) ** 2

            fidelities = [_alpha_least_square(p_sim[i], p_data[i]) for i in range(num_p)]

            cost = -np.sum(fidelities)

            err, x_vals, y_vals = pauli_error_fit(
                np.asarray(num_cycle_range)[0:num_p],
                np.asarray(fidelities),
                add_offset=False,
            )

            return err, fidelities, x_vals, y_vals, cost

        def cost_function(angle_shifts: np.ndarray) -> float:
            # Accepts shifts in a variable number of FSIM angles and outputs a cost function (i.e.
            # XEB fidelity). If the sqrt-iSWAP is the gate, shifts in delta_plus,
            # delta_minus_off_diag and delta_minus_diag are specified. If iSWAP is the gate,
            # shifts in delta_plus and delta_minus_off_diag are specified. If CZ is the gate,
            # shifts in delta_plus and delta_minus_diag are specified. In other cases, shifts in
            # all 5 angles are specified. The unspecified angles are set to have zero shifts from
            # their initial values.

            if gate_to_fit == "sqrt-iswap":
                full_shifts = np.zeros(5, dtype=float)
                full_shifts[1:4] = angle_shifts
            elif gate_to_fit == "iswap":
                full_shifts = np.zeros(5, dtype=float)
                full_shifts[1:3] = angle_shifts
            elif gate_to_fit == "cz" or gate_to_fit == "composite-cz":
                full_shifts = np.zeros(5, dtype=float)
                full_shifts[1] = angle_shifts[0]
                full_shifts[3] = angle_shifts[1]
            else:
                full_shifts = angle_shifts
            _, _, _, _, cost = xeb_fidelity(full_shifts, num_p=num_points)
            return cost

        sp_purities = [_speckle_purity(p_data[i]) ** 0.5 for i in range(len(num_cycle_range))]
        err_p, x_vals_p, y_vals_p = pauli_error_fit(
            np.asarray(num_cycle_range), np.asarray(sp_purities), add_offset=True
        )
        purity_errors[(q0, q1)] = err_p

        err_0, f_vals_0, x_fitted_0, y_fitted_0, _ = xeb_fidelity(
            np.zeros(5), num_p=len(num_cycle_range)
        )
        final_errors_unoptimized[(q0, q1)] = err_0

        # Set up initial guesses on the relevant FSIM angles according to the ideal gate. See
        # comments in cost_function. All angles are allowed to shift up to +/- 1 rad from their
        # ideal (initial guess) values.
        err_min = 1.0
        soln_vec = np.zeros(5)
        if gate_to_fit == "sqrt-iswap":
            init_guess = np.zeros(3)
            bounds = (np.ones(3) * -1.0, np.ones(3) * 1.0)
        elif gate_to_fit == "iswap" or gate_to_fit == "cz" or gate_to_fit == "composite-cz":
            init_guess = np.zeros(2)
            bounds = (np.ones(2) * -1.0, np.ones(2) * 1.0)
        else:
            init_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
            bounds = (np.ones(5) * -1.0, np.ones(5) * 1.0)

        for _ in range(num_restarts):
            res = pybobyqa.solve(
                cost_function, init_guess, maxfun=3000, bounds=bounds, rhoend=1e-11
            )

            # Randomize the initial values for the relevant FSIM angles.
            if gate_to_fit == "sqrt-iswap":
                init_guess = np.random.uniform(-0.3, 0.3, 3)
            elif gate_to_fit == "iswap" or gate_to_fit == "cz" or gate_to_fit == "composite-cz":
                init_guess = np.random.uniform(-0.3, 0.3, 2)
            else:
                init_guess = np.random.uniform(-0.2, 0.2, 5)
                init_guess[0] = 0.0
                init_guess[4] = 0.1

            if res.f < err_min:
                err_min = res.f
                if gate_to_fit == "sqrt-iswap":
                    soln_vec = np.zeros(5)
                    soln_vec[1:4] = np.asarray(res.x)
                elif gate_to_fit == "iswap":
                    soln_vec = np.zeros(5)
                    soln_vec[1:3] = np.asarray(res.x)
                elif gate_to_fit == "cz" or gate_to_fit == "composite-cz":
                    soln_vec = np.zeros(5)
                    soln_vec[1] = np.asarray(res.x)[0]
                    soln_vec[3] = np.asarray(res.x)[1]
                else:
                    soln_vec = np.asarray(res.x)

        err_1, f_vals_1, x_fitted_1, y_fitted_1, _ = xeb_fidelity(
            soln_vec, num_p=len(num_cycle_range)
        )
        final_errors_optimized[(q0, q1)] = err_1

        delta_angles[(q0, q1)] = {a: soln_vec[i] for i, a in enumerate(_fsim_angle_labels)}

        new_angles = fsim_angles.copy()
        for k, v in new_angles.items():
            new_angles[k] += delta_angles[(q0, q1)][k]

        fitted_angles[(q0, q1)] = new_angles

        q_0 = cirq.GridQubit(*q0)
        q_1 = cirq.GridQubit(*q1)

        gate_list = generic_fsim_gate(new_angles, (q_0, q_1))
        circuit_fitted = cirq.Circuit(gate_list)
        fitted_gates[(q0, q1)] = circuit_fitted

        # Use the fitted FSIM to set up the virtual-Z gates that are needed to cancel out the
        # shifts in the SQ phases (i.e. delta angles).
        corrected_angles = new_angles.copy()
        corrected_angles["delta_plus"] *= -1.0
        corrected_angles["delta_minus_off_diag"] *= -1.0
        corrected_angles["delta_minus_diag"] *= -1.0
        corrected_angles["theta"] = fsim_angles["theta"]
        corrected_angles["phi"] = fsim_angles["phi"]
        gate_list_corrected = generic_fsim_gate(corrected_angles, (q_0, q_1))

        if gate_to_fit == "composite-cz":
            circuit_corrected = cirq.Circuit(gate_list_corrected[0:2])
            circuit_corrected.append(cz_to_sqrt_iswap(q_0, q_1))
            circuit_corrected.append(cirq.Moment(gate_list_corrected[-2:]))
        else:
            circuit_corrected = cirq.Circuit(gate_list_corrected)

        correction_gates[(q0, q1)] = circuit_corrected

        raw_data[(q0, q1)] = XEBData(
            fidelity_optimized=(np.asarray(num_cycle_range), np.asarray(f_vals_1)),
            fidelity_optimized_fit=(x_fitted_1, y_fitted_1),
            fidelity_unoptimized=(np.asarray(num_cycle_range), np.asarray(f_vals_0)),
            fidelity_unoptimized_fit=(x_fitted_0, y_fitted_0),
            purity=(np.asarray(num_cycle_range), np.asarray(sp_purities)),
            purity_fit=(x_vals_p, y_vals_p),
        )

    return ParallelXEBResults(
        fitted_gates=fitted_gates,
        correction_gates=correction_gates,
        fitted_angles=fitted_angles,
        final_errors_optimized=final_errors_optimized,
        final_errors_unoptimized=final_errors_unoptimized,
        purity_errors=purity_errors,
        raw_data=raw_data,
    )


def _random_rotations(
    qubits: Sequence[cirq.GridQubit],
    num_layers: int,
    rand_seed: Optional[int] = None,
    rand_nums: Optional[np.ndarray] = None,
    light_cones: Optional[List[List[Set[cirq.GridQubit]]]] = None,
    echo_indices: Optional[np.ndarray] = None,
    z_rotations_only: bool = False,
) -> Tuple[List[List[cirq.OP_TREE]], np.ndarray]:
    """Generate random single-qubit rotations and group them into different circuit layers."""
    num_qubits = len(qubits)

    random_state = cirq.value.parse_random_state(rand_seed)

    if rand_nums is None:
        if z_rotations_only:
            rand_nums = random_state.uniform(-1, 1, (num_qubits, num_layers))
        else:
            rand_nums = random_state.choice(8, (num_qubits, num_layers))

    single_q_layers = []  # type: List[List[cirq.OP_TREE]]

    for i in range(num_layers):
        op_seq = []
        for j in range(num_qubits):
            gate_choice = 0
            if light_cones is not None:
                if len(light_cones) == 1:
                    if qubits[j] not in light_cones[0][i]:
                        gate_choice = 1
                elif len(light_cones) == 2:
                    if qubits[j] not in light_cones[1][i]:
                        gate_choice = 2
                    elif qubits[j] not in light_cones[0][i]:
                        gate_choice = 1
            if gate_choice == 0:
                if z_rotations_only:
                    op_seq.append(cirq.Z(qubits[j]) ** rand_nums[j, i])
                else:
                    op_seq.append(_rot_ops[rand_nums[j, i]](qubits[j]))

            elif gate_choice == 1:
                if echo_indices is None:
                    op_seq.append(cirq.Y(qubits[j]))
                else:
                    if echo_indices[j, i] > 0:
                        op_seq.append(_spin_echo_gates(echo_indices[j, i])(qubits[j]))
            else:
                continue
        single_q_layers.append(op_seq)
    return single_q_layers, rand_nums


def _alpha_least_square(probs_exp: np.ndarray, probs_data: np.ndarray) -> float:
    """Compare an ideal and an experimental probability distribution and compute their
    cross-entropy fidelity.
    """
    if probs_exp.shape != probs_data.shape:
        raise ValueError("probs_exp and probs_data must have the same shape")
    num_trials, num_states = probs_exp.shape
    p_exp = np.maximum(probs_exp, np.zeros_like(probs_exp) + 1e-22)
    p_data = np.maximum(probs_data, np.zeros_like(probs_data))
    nominator = 0.0
    denominator = 0.0
    p_uni = 1.0 / num_states
    for i in range(num_trials):
        s_incoherent = -np.sum(p_uni * np.log(p_exp[i, :]))
        s_expected = -np.sum(p_exp[i, :] * np.log(p_exp[i, :]))
        s_meas = -np.sum(p_data[i, :] * np.log(p_exp[i, :]))
        delta_h_meas = float(s_incoherent - s_meas)
        delta_h = float(s_incoherent - s_expected)
        nominator += delta_h_meas * delta_h
        denominator += delta_h ** 2
    return nominator / denominator


def _speckle_purity(probs_data: np.ndarray) -> float:
    """Compute the speckle purity of a probability distribution"""
    d = 4
    return np.var(probs_data) * d ** 2 * (d + 1) / (d - 1)


def _pairwise_xeb_probabilities(
    all_qubits: List[Tuple[int, int]],
    num_cycle_range: Sequence[int],
    measured_bits: List[List[List[np.ndarray]]],
    scrambling_gates: List[List[np.ndarray]],
    interaction_sequence: Optional[
        List[Set[Tuple[Tuple[float, float], Tuple[float, float]]]]
    ] = None,
) -> Tuple[
    Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[np.ndarray]],
    Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[np.ndarray]],
]:
    """Computes the probability distributions of each qubit pair in a parallel XEB experiment.

    Args:
        all_qubits: List of qubits involved in parallel XEB, specified as (col, row) tuples.
        num_cycle_range: Different numbers of circuit cycles used in parallel XEB.
        measured_bits: The experimental bit-strings stored in a nested list. The first dimension
            of the nested list represents different configurations (e.g. how the two-qubit gates
            are applied) used in parallel XEB. The second dimension represents different trials
            (i.e. random circuit instances) used in XEB. The third dimension represents the
            different numbers of cycles and must be the same as len(num_cycle_range). Each
            np.ndarray has dimension M x N, where M is the number of repetitions (stats) for each
            circuit and N is the number of qubits involved.
        scrambling_gates: The random circuit indices specified as integers between 0 and 7. See
            the documentation of build_xeb_circuits for details. The first dimension of the
            nested list represents the different configurations and must be the same as the first
            dimension of measured_bits. The second dimension represents the different trials and
            must be the same as the second dimension of measured_bits.
        interaction_sequence: The pairs of qubits with FSIM applied for each configuration. Must
            be the same as len(measured_bits).

    Returns:
        p_data_all: Keys are qubit pairs. Each value is a list (with length =
            len(num_cycle_range)) of np.array. The rows of the array are of length 4 and
            represent the measured probabilities of two-qubit basis states. Each row represents
            the result from a different trial (circuit instance).
        sq_gates: Keys are qubit pairs. Each value is a list (with length = number of circuit
            instances) of np.array. Each array contains the indices (integers between 0 and 7)
            for the random SQ gates relevant to the given qubit pair.
    """
    num_trials = len(measured_bits[0])
    qubits = [cirq.GridQubit(*idx) for idx in all_qubits]

    if interaction_sequence is None:
        int_layers = _default_interaction_sequence(qubits)
    else:
        int_layers = [
            {(cirq.GridQubit(i, j), cirq.GridQubit(k, l)) for ((i, j), (k, l)) in layer}
            for layer in interaction_sequence
        ]

    p_data_all = {}
    sq_gates = {}
    for l, qubit_set in enumerate(int_layers):
        qubit_pairs = [
            ((q_s[0].row, q_s[0].col), (q_s[1].row, q_s[1].col)) for q_s in int_layers[l]
        ]
        p_data = {
            q_s: [np.zeros((num_trials, 4)) for _ in range(len(num_cycle_range))]
            for q_s in qubit_pairs
        }
        for (q0, q1) in qubit_pairs:
            idx_0, idx_1 = all_qubits.index(q0), all_qubits.index(q1)
            sq_gates[(q0, q1)] = [
                scrambling_gates[l][k][[idx_0, idx_1], :] for k in range(num_trials)
            ]

        for i in range(num_trials):
            for j in range(len(num_cycle_range)):
                bits = measured_bits[l][i][j]
                for q_s in qubit_pairs:
                    p_data[q_s][j][i, :] = bits_to_probabilities(all_qubits, q_s, bits)
        p_data_all = {**p_data_all, **p_data}
    return p_data_all, sq_gates


def _spin_echo_gates(idx: int) -> cirq.ops:
    """Outputs one of 4 single-qubit pi rotations which is used for spin echoes."""
    pi_pulses = [
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0),
        cirq.PhasedXPowGate(phase_exponent=0.5, exponent=1.0),
        cirq.PhasedXPowGate(phase_exponent=1.0, exponent=1.0),
        cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=1.0),
    ]
    return pi_pulses[idx - 1]

def _default_interaction_sequence(
    qubits: Sequence[cirq.GridQubit],
) -> List[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]]:
    qubit_dict = {(qubit.row, qubit.col): qubit for qubit in qubits}
    qubit_locs = set(qubit_dict)
    num_rows = max([q.row for q in qubits]) + 1
    num_cols = max([q.col for q in qubits]) + 1

    l_s: List[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]] = [set() for _ in range(4)]
    for i in range(num_rows):
        for j in range(num_cols - 1):
            if (i, j) in qubit_locs and (i, j + 1) in qubit_locs:
                l_s[j % 2 * 2].add((qubit_dict[(i, j)], qubit_dict[(i, j + 1)]))

    for i in range(num_rows - 1):
        for j in range(num_cols):
            if (i, j) in qubit_locs and (i + 1, j) in qubit_locs:
                l_s[i % 2 * 2 + 1].add((qubit_dict[(i, j)], qubit_dict[(i + 1, j)]))

    l_final: List[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]] = []
    for gate_set in l_s:
        if len(gate_set) != 0:
            l_final.append(gate_set)

    return l_final

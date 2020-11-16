from typing import Sequence, List, Set, Tuple, Dict, Union

import cirq
import numpy
import pybobyqa
from matplotlib import pyplot

from recirq.otoc.utils import bits_to_probabilities, angles_to_fsim, \
    pauli_error_fit, generic_fsim_gate

_rot_ops = [
    cirq.X ** 0.5, cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5),
    cirq.Y ** 0.5, cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.5),
    cirq.X ** -0.5, cirq.PhasedXPowGate(phase_exponent=-0.75, exponent=0.5),
    cirq.Y ** -0.5, cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=0.5),
]
_rot_mats = [cirq.unitary(r) for r in _rot_ops]
_fsim_angle_labels = ['theta', 'delta_plus', 'delta_minus_off_diag',
                      'delta_minus_diag', 'phi']


def default_interaction_sequence(
        qubits: Sequence[cirq.GridQubit]
) -> List[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]]:
    """Group qubits on a grid into sets of pairs.

    The qubits are restricted to be physically on a square grid with distinct
    row and column indices (not every node of the grid needs to have a
    qubit). They are grouped into a maximum of 4 sets, each containing pairs
    of qubits that do not intersect with each other. The method for grouping
    is as follows:

    The first set groups qubits (i, j) and (i, j + 1) where i is any integer
    and j is an even integer. The second set groups qubits (i, j) and
    (i + 1, j) where i is an even integer and j is any integer. The third
    set groups qubits (i, j) and (i, j + 1) where i is any integer and j is
    an odd integer. The last set groups qubits (i, j) and (i + 1, j) where i
    is an odd integer and j is any integer.

    After the sets are built as above, any empty set is ejected.:

                 Cycle 1:                            Cycle 2:
        q00 ── q01    q02 ── q03            q00    q01    q02    q03
                                             |      |      |      |
        q10 ── q11    q12 ── q13            q10    q11    q12    q13

        q20 ── q21    q22 ── q23            q20    q21    q22    q23
                                             |      |      |      |
        q30 ── q31    q32 ── q33            q30    q31    q32    q33

                  Cycle 3:                           Cycle 4:
        q00    q01 ── q02    q03            q00    q01    q02    q03

        q10    q11 ── q12    q13            q10    q11    q12    q13
                                             |      |      |      |
        q20    q21 ── q22    q23            q20    q21    q22    q23

        q30    q31 ── q32    q33            q30    q31    q32    q33

    Args:
        qubits: A sequence of cirq.GridQubit objects representing the qubits
            to be grouped.

    Returns:
        A list of at most 4 sets of GridQubit tuples.
    """
    qubit_dict = {(qubit.row, qubit.col): qubit for qubit in qubits}
    qubit_locs = set(qubit_dict)
    num_rows = max([q.row for q in qubits]) + 1
    num_cols = max([q.col for q in qubits]) + 1

    l_s = [set() for _ in range(4)
           ]  # type: List[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]]
    for i in range(num_rows):
        for j in range(num_cols - 1):
            if (i, j) in qubit_locs and (i, j + 1) in qubit_locs:
                l_s[j % 2 * 2].add((qubit_dict[(i, j)], qubit_dict[(i, j + 1)]))

    for i in range(num_rows - 1):
        for j in range(num_cols):
            if (i, j) in qubit_locs and (i + 1, j) in qubit_locs:
                l_s[i % 2 * 2 + 1].add(
                    (qubit_dict[(i, j)], qubit_dict[(i + 1, j)]))

    l_final = []  # type: List[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]]
    for gate_set in l_s:
        if len(gate_set) != 0:
            l_final.append(gate_set)

    return l_final


def build_xeb_circuits(qubits: Sequence[cirq.GridQubit],
                       cycles: Sequence[int],
                       benchmark_ops: Sequence[Union[
                           cirq.Moment, Sequence[cirq.Moment]]] = None,
                       random_seed: int = None,
                       sq_rand_nums: numpy.ndarray = None,
                       reverse: bool = False,
                       z_only: bool = False,
                       ancilla: cirq.GridQubit = None,
                       cycles_per_echo: int = None,
                       light_cones: List[List[Set[cirq.GridQubit]]] = None,
                       echo_indices: numpy.ndarray = None,
                       ) -> Tuple[List[cirq.Circuit], numpy.ndarray]:
    """Builds random circuits for cross entropy benchmarking (XEB).

    A list of cirq.Circuits of varying lengths are generated, which are made
    of random single-qubit gates and optional two-qubit gates.

    Args:
        qubits: The qubits to be involved in XEB.
        cycles: The different numbers of cycles the random circuits will have.
        benchmark_ops: The operations to be inserted between random
            single-qubit gates. They can be one or more cirq.Moment objects,
            or None (in which case no operation will be inserted between the
            random single-qubit gates).
        random_seed: The random seed for the single-qubit gates. If
            unspecified, no random seed will be used.
        sq_rand_nums: The random numbers representing the single-qubit gates.
            They must be integers from 0 to 7 if the z_only is False,
            and floats between -1 and 1 if z_only is True. The dimension of
            sq_rand_nums should be len(qubits) by max(cycles). If
            unspecified, the gates will be generated in-situ at random.
        reverse: If True, benchmark_ops will be applied before the random
            single-qubit gates in each cycle. Otherwise, it will be applied
            after the random single-qubit gates.
        z_only: Whether the single-qubit gates are to be random \pi/2
            rotations around axes on the equatorial plane of the Bloch sphere
            (z_only = False), or random rotations around the z-axis (z_only =
            True). In the former case, the axes of rotations will be chosen
            randomly from 8 evenly spaced axes (\pi/4, \pi/2 ... 7\pi/4
            radians from the x-axis). In the latter case, the angles of
            rotation will be any random value between -\pi and \pi.
        ancilla: If specified, an additional qubit will be included in the
            circuit which does not interact with the other qubits and only
            has spin-echo pulses applied to itself.
        cycles_per_echo: How often a spin-echo (Y gate) gate is to be applied
            to the ancilla qubit. For example, if the value is 2, a Y gate
            will be applied every other cycle.
        light_cones: A list of length 1 or 2, each specifying a lightcone
            correponding to a list of sets of qubits with the same length as
            max(cycles). For each cycle, single-qubit gates outside the first
            lightcone are either removed or replaced with a spin-echo pulse.
            Single-qubit gates outside the second lightcone, if specified,
            are always removed.
        echo_indices: An array with the same dimension as sq_rand_nums and
            random integer values of 1, 2, 3 or 4. They specify the spin-echo
            pulses applied to qubits outside the first lightcone, which can
            be +/-X or +/-Y gates.

    Returns:
        A list of random circuits, each containing a specified number of cycles.
    """
    if light_cones is not None:
        if len(light_cones) > 2:
            raise ValueError('light_cones may only have length 1 or 2')

    if benchmark_ops is not None:
        num_d = len(benchmark_ops)
    else:
        num_d = 0
    max_cycles = max(cycles)

    single_rots, indices = _random_rotations(
        qubits, max_cycles, random_seed, sq_rand_nums, light_cones,
        echo_indices, z_rotations_only=z_only)

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
    return all_circuits, indices


def parallel_xeb_fidelities(
        all_qubits: List[Tuple[int, int]],
        num_cycle_range: Sequence[int],
        measured_bits: List[List[List[numpy.ndarray]]],
        scrambling_gates: List[List[numpy.ndarray]],
        fsim_angles: Dict[str, float],
        interaction_sequence: List[
            Set[Tuple[Tuple[float, float], Tuple[float, float]]]] = None,
        gate_to_fit: str = 'iswap',
        num_restarts: int = 3,
        num_points: int = 8,
        plot_individual_traces: bool = False,
        plot_histograms: bool = False,
        save_directory: str = None
) -> Tuple[Dict[Tuple[Tuple[int, int], Tuple[int, int]], cirq.Circuit],
           Dict[Tuple[Tuple[int, int], Tuple[int, int]], cirq.Circuit],
           Dict[Tuple[Tuple[int, int], Tuple[int, int]], Dict[str, float]],
           Dict[Tuple[Tuple[int, int], Tuple[int, int]], float],
           Dict[Tuple[Tuple[int, int], Tuple[int, int]], float]]:
    """Computes and optimizes cycle fidelities from parallel XEB data.

    Args:
        all_qubits: List of qubits involved in a parallel XEB experiment,
            specified using their (row, col) locations.
        num_cycle_range: The different numbers of cycles in the random circuits.
        measured_bits: The experimental bit-strings stored in a nested list.
            The first dimension of the nested list represents different
            configurations (e.g. how the two-qubit gates are applied) used in
            parallel XEB. The second dimension represents different trials (
            i.e. random circuit instances) used in XEB. The third dimension
            represents the different numbers of cycles and must be the same
            as len(num_cycle_range). Each np.ndarray has dimension M x N,
            where M is the number of repetitions (stats) for each circuit and N
            is the number of qubits involved.
        scrambling_gates: The random circuit indices specified as integers
            between 0 and 7. See the documentation of build_xeb_circuits for
            details. The first dimension of the nested list represents the
            different configurations and must be the same as the first
            dimension of measured_bits. The second dimension represents
            the different trials and must be the same as the second dimension
            of measured_bits.
        fsim_angles: An initial guess for the five FSIM angles for each qubit
            pair.
        interaction_sequence: The pairs of qubits with FSIM applied for each
            configuration. Must be the same as len(measured_bits).
        gate_to_fit: Can be either 'iswap', 'sqrt-iswap', 'cz' or any other
            string. Determines the FSIM angles that will be changed from their
            initial guess values to optimize the XEB fidelity of each qubit
            pair. For 'iswap', only 'delta_plus' and 'delta_minus_off_diag'
            are changed. For 'sqrt-iswap', 'delta_plus',
            'delta_minus_off_diag' and 'delta_minus_diag' are changed. For
            'cz', only 'delta_plus' and 'delta_minus_diag' are changed. For
            any other string, all five angles are changed.
        num_restarts: Number of restarts with different random initial guesses.
        num_points: The total number of XEB fidelities to be used in the
            cost function for optimization. Default is 8, such that the cost
            function is the sum of the XEB fidelities for the first 8 numbers
            of cycles in num_cycle_range.
        plot_individual_traces: Whether to plot the XEB fidelities along with
            the fitting results for each qubit pair.
        plot_histograms: Whether to plot the histograms of cycle fidelities
            and changes in FSIM angles after fitting for all qubit pairs.
        save_directory: Directory to which the plots are to be saved. If
            unspecified, the plots will not be saved.

    Returns:
        fitted_gates: A dictionary with qubit pairs as keys and optimized
            FSIM unitaries, represented by cirq.Circuit objects, as values.
        correction_gates: Same as fitted_gates, but with all Z rotations
            reversed in signs.
        fitted_angles: A dictionary with qubit pairs as keys and optimized
            FSIM unitaries as values. Here the FSIM unitaries are represented
            as a dictionaries with the names of the FSIM phases as keys and
            their fitted values as values.
        final_errors_optimized: A dictionary with qubit pairs as keys and
            their cycle errors after fitting as values.
        final_errors_unoptimized: A dictionary with qubit pairs as keys and
            their cycle errors before fitting as values.
    """
    num_trials = len(measured_bits[0])
    p_data_all, sq_gates = _pairwise_xeb_probabilities(
        all_qubits, num_cycle_range, measured_bits, scrambling_gates,
        interaction_sequence)
    final_errors_unoptimized = {}
    final_errors_optimized = {}
    delta_angles = {}
    purity_errors = {}
    fitted_gates = {}
    fitted_angles = {}
    correction_gates = {}

    for (q0, q1), p_data in p_data_all.items():
        print('Fitting qubits {} and {}'.format(q0, q1))

        def xeb_fidelity(angle_shifts: numpy.ndarray, num_p: int
                         ) -> Tuple[float, List[float], numpy.ndarray,
                                    numpy.ndarray, float]:
            new_angles = fsim_angles.copy()
            for i, angle_name in enumerate(_fsim_angle_labels):
                new_angles[angle_name] += angle_shifts[i]
            fsim_mat = angles_to_fsim(**new_angles)

            max_cycles = num_cycle_range[num_p - 1]

            p_sim = [numpy.zeros((num_trials, 4)) for _ in
                     range(max_cycles)]
            for i in range(num_trials):
                unitary = numpy.identity(4, dtype=complex)
                for j in range(max_cycles):
                    mat_0 = _rot_mats[sq_gates[(q0, q1)][i][0, j]]
                    mat_1 = _rot_mats[sq_gates[(q0, q1)][i][1, j]]
                    unitary = numpy.kron(mat_0, mat_1).dot(unitary)
                    unitary = fsim_mat.dot(unitary)
                    if j + 1 in num_cycle_range:
                        idx = num_cycle_range.index(j + 1)
                        p_sim[idx][i, :] = numpy.abs(unitary[:, 0]) ** 2

            fidelities = [_alpha_least_square(p_sim[i], p_data[i]) for i in
                          range(num_p)]

            cost = -numpy.sum(fidelities)

            err, x_vals, y_vals = pauli_error_fit(
                numpy.asarray(num_cycle_range)[0:num_p],
                numpy.asarray(fidelities), add_offset=False)

            return err, fidelities, x_vals, y_vals, cost

        def cost_function(angle_shifts: numpy.ndarray) -> float:
            if gate_to_fit == 'sqrt-iswap':
                full_shifts = numpy.zeros(5, dtype=float)
                full_shifts[1:4] = angle_shifts
            elif gate_to_fit == 'iswap':
                full_shifts = numpy.zeros(5, dtype=float)
                full_shifts[1:3] = angle_shifts
            elif gate_to_fit == 'cz':
                full_shifts = numpy.zeros(5, dtype=float)
                full_shifts[1] = angle_shifts[0]
                full_shifts[3] = angle_shifts[1]
            else:
                full_shifts = angle_shifts
            _, _, _, _, cost = xeb_fidelity(full_shifts, num_p=num_points)
            return cost

        sqrt_purities = [_speckle_purity(p_data[i]) ** 0.5 for i in
                         range(len(num_cycle_range))]
        err_p, x_vals_p, y_vals_p = pauli_error_fit(
            numpy.asarray(num_cycle_range), numpy.asarray(sqrt_purities),
            add_offset=True)
        purity_errors[(q0, q1)] = err_p

        err_0, f_vals_0, x_fitted_0, y_fitted_0, _ = xeb_fidelity(
            numpy.zeros(5), num_p=len(num_cycle_range))
        final_errors_unoptimized[(q0, q1)] = err_0

        err_min = 1.0
        soln_vec = numpy.zeros(5)
        if gate_to_fit == 'sqrt-iswap':
            init_guess = numpy.zeros(3)
            bounds = (numpy.ones(3) * -1.0, numpy.ones(3) * 1.0)
        elif gate_to_fit == 'iswap' or gate_to_fit == 'cz':
            init_guess = numpy.zeros(2)
            bounds = (numpy.ones(2) * -1.0, numpy.ones(2) * 1.0)
        else:
            init_guess = numpy.array([0.0, 0.0, 0.0, 0.0, 0.1])
            bounds = (numpy.ones(5) * -0.6, numpy.ones(5) * 0.6)

        for _ in range(num_restarts):
            res = pybobyqa.solve(
                cost_function, init_guess, maxfun=3000, bounds=bounds,
                rhoend=1e-11)

            if gate_to_fit == 'sqrt-iswap':
                init_guess = numpy.random.uniform(-0.3, 0.3, 3)
            elif gate_to_fit == 'iswap' or gate_to_fit == 'cz':
                init_guess = numpy.random.uniform(-0.3, 0.3, 2)
            else:
                init_guess = numpy.random.uniform(-0.2, 0.2, 5)
                init_guess[0] = 0.0
                init_guess[4] = 0.1

            if res.f < err_min:
                err_min = res.f
                if gate_to_fit == 'sqrt-iswap':
                    soln_vec = numpy.zeros(5)
                    soln_vec[1:4] = numpy.asarray(res.x)
                elif gate_to_fit == 'iswap':
                    soln_vec = numpy.zeros(5)
                    soln_vec[1:3] = numpy.asarray(res.x)
                elif gate_to_fit == 'cz':
                    soln_vec = numpy.zeros(5)
                    soln_vec[1] = numpy.asarray(res.x)[0]
                    soln_vec[3] = numpy.asarray(res.x)[1]
                else:
                    soln_vec = numpy.asarray(res.x)

        err_1, f_vals_1, x_fitted_1, y_fitted_1, _ = xeb_fidelity(
            soln_vec, num_p=len(num_cycle_range))
        final_errors_optimized[(q0, q1)] = err_1

        delta_angles[(q0, q1)] = {a: soln_vec[i] for i, a in
                                  enumerate(_fsim_angle_labels)}

        new_angles = fsim_angles.copy()
        for k, v in new_angles.items():
            new_angles[k] += delta_angles[(q0, q1)][k]

        fitted_angles[(q0, q1)] = new_angles

        q_0 = cirq.GridQubit(*q0)
        q_1 = cirq.GridQubit(*q1)

        gate_list = generic_fsim_gate(new_angles, (q_0, q_1))
        circuit_fitted = cirq.Circuit(gate_list)
        fitted_gates[(q0, q1)] = circuit_fitted

        corrected_angles = new_angles.copy()
        corrected_angles['delta_plus'] *= -1.0
        corrected_angles['delta_minus_off_diag'] *= -1.0
        corrected_angles['delta_minus_diag'] *= -1.0
        corrected_angles['theta'] = fsim_angles['theta']
        corrected_angles['phi'] = fsim_angles['phi']
        gate_list_corrected = generic_fsim_gate(corrected_angles, (q_0, q_1))
        circuit_corrected = cirq.Circuit(gate_list_corrected)
        correction_gates[(q0, q1)] = circuit_corrected

        fig = pyplot.figure()
        pyplot.plot(
            num_cycle_range, f_vals_0, 'ro', figure=fig,
            label=r'{} and {}, unoptimized [$r_p$ = {}]'.format(
                q0, q1, err_0.__round__(5)))
        pyplot.plot(
            num_cycle_range, f_vals_1, 'bo', figure=fig,
            label=r'{} and {}, optimized [$r_p$ = {}]'.format(
                q0, q1, err_1.__round__(5)))
        pyplot.plot(
            num_cycle_range, sqrt_purities, 'go', figure=fig,
            label=r'{} and {}, purity error = {}'.format(
                q0, q1, err_p.__round__(5)))
        pyplot.plot(x_fitted_0, y_fitted_0, 'r--')
        pyplot.plot(x_fitted_1, y_fitted_1, 'b--')
        pyplot.plot(x_vals_p, y_vals_p, 'g--')
        pyplot.legend()
        pyplot.xlabel('Number of Cycles')
        pyplot.ylabel(r'XEB Fidelity')

        if save_directory is not None:
            fig.savefig(save_directory +
                        '/xeb_q{}_{}_q{}_{}'.format(
                            q0[0], q0[1], q1[0], q1[1]))

        if not plot_individual_traces:
            pyplot.close(fig)

    num_pairs = len(final_errors_optimized)
    pair_pos = numpy.linspace(0, 1, num_pairs)

    fig_0 = pyplot.figure()
    pyplot.plot(sorted(final_errors_unoptimized.values()), pair_pos,
                figure=fig_0, label='Unoptimized Unitaries')
    pyplot.plot(sorted(final_errors_optimized.values()), pair_pos,
                figure=fig_0, label='Optimized Unitaries')
    pyplot.plot(sorted(purity_errors.values()), pair_pos,
                figure=fig_0, label='Purity Errors')
    pyplot.xlabel(r'Pauli Error Rate, $r_p$')
    pyplot.ylabel(r'Integrated Histogram')
    pyplot.legend()

    fig_1 = pyplot.figure()
    for label in _fsim_angle_labels:
        shifts = [a[label] for a in delta_angles.values()]
        pyplot.plot(sorted(shifts), pair_pos, figure=fig_1, label=label)
    pyplot.xlabel(r'FSIM Angle Error (Radian)')
    pyplot.ylabel(r'Integrated Histogram')
    pyplot.legend()

    if not plot_histograms:
        pyplot.close(fig_0)
        pyplot.close(fig_1)

    if save_directory is not None:
        fig_0.savefig(save_directory + 'pauli_error_histogram')
        fig_1.savefig(save_directory + 'angle_shift_histogram')

    return (fitted_gates, correction_gates, fitted_angles,
            final_errors_optimized, final_errors_unoptimized)


def _random_rotations(qubits: Sequence[cirq.GridQubit],
                      num_layers: int,
                      rand_seed: int = None,
                      rand_nums: numpy.ndarray = None,
                      light_cones: List[List[Set[cirq.GridQubit]]] = None,
                      echo_indices: numpy.ndarray = None,
                      z_rotations_only: bool = False
                      ) -> Tuple[List[List[cirq.OP_TREE]], numpy.ndarray]:
    num_qubits = len(qubits)

    if rand_seed is not None:
        numpy.random.seed(rand_seed)

    if rand_nums is None:
        if z_rotations_only:
            rand_nums = numpy.random.uniform(-1, 1, (num_qubits, num_layers))
        else:
            rand_nums = numpy.random.choice(8, (num_qubits, num_layers))

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
                        op_seq.append(
                            _spin_echo_gates(echo_indices[j, i])(qubits[j]))
            else:
                continue
        single_q_layers.append(op_seq)
    return single_q_layers, rand_nums


def _alpha_least_square(probs_exp: numpy.ndarray, probs_data: numpy.ndarray
                        ) -> float:
    if probs_exp.shape != probs_data.shape:
        raise ValueError('probs_exp and probs_data must have the same shape')
    num_trials, num_states = probs_exp.shape
    p_exp = (numpy.maximum(probs_exp, numpy.zeros_like(probs_exp) + 1e-22))
    p_data = (numpy.maximum(probs_data, numpy.zeros_like(probs_data)))
    nominator = 0.0
    denominator = 0.0
    p_uni = 1.0 / num_states
    for i in range(num_trials):
        s_incoherent = -numpy.sum(p_uni * numpy.log(p_exp[i, :]))
        s_expected = -numpy.sum(p_exp[i, :] * numpy.log(p_exp[i, :]))
        s_meas = -numpy.sum(p_data[i, :] * numpy.log(p_exp[i, :]))
        delta_h_meas = float(s_incoherent - s_meas)
        delta_h = float(s_incoherent - s_expected)
        nominator += delta_h_meas * delta_h
        denominator += delta_h ** 2
    return nominator / denominator


def _speckle_purity(probs_data: numpy.ndarray) -> float:
    d = 4
    return numpy.var(probs_data) * d ** 2 * (d + 1) / (d - 1)


def _pairwise_xeb_probabilities(
        all_qubits: List[Tuple[int, int]],
        num_cycle_range: Sequence[int],
        measured_bits: List[List[List[numpy.ndarray]]],
        scrambling_gates: List[List[numpy.ndarray]],
        interaction_sequence: List[Set[Tuple[
            Tuple[float, float], Tuple[float, float]]]] = None,
) -> Tuple[Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[numpy.ndarray]],
           Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[numpy.ndarray]]]:
    num_trials = len(measured_bits[0])
    qubits = [cirq.GridQubit(*idx) for idx in all_qubits]

    if interaction_sequence is None:
        int_layers = default_interaction_sequence(qubits)
    else:
        int_layers = [{(cirq.GridQubit(i, j), cirq.GridQubit(k, l)) for
                       ((i, j), (k, l)) in layer} for layer in
                      interaction_sequence]

    p_data_all = {}
    sq_gates = {}
    for l, qubit_set in enumerate(int_layers):
        qubit_pairs = [((q_s[0].row, q_s[0].col), (q_s[1].row, q_s[1].col))
                       for q_s in int_layers[l]]
        p_data = {q_s: [numpy.zeros((num_trials, 4)) for _ in
                        range(len(num_cycle_range))] for q_s in qubit_pairs}
        for (q0, q1) in qubit_pairs:
            idx_0, idx_1 = all_qubits.index(q0), all_qubits.index(q1)
            sq_gates[(q0, q1)] = [scrambling_gates[l][k][[idx_0, idx_1], :] for
                                  k in range(num_trials)]

        for i in range(num_trials):
            for j in range(len(num_cycle_range)):
                bits = measured_bits[l][i][j]
                for q_s in qubit_pairs:
                    p_data[q_s][j][i, :] = bits_to_probabilities(
                        all_qubits, q_s, bits)
        p_data_all = {**p_data_all, **p_data}
    return p_data_all, sq_gates


def _spin_echo_gates(idx: int) -> cirq.ops:
    pi_pulses = [
        cirq.PhasedXPowGate(phase_exponent=0.0, exponent=1.0),
        cirq.PhasedXPowGate(phase_exponent=0.5, exponent=1.0),
        cirq.PhasedXPowGate(phase_exponent=1.0, exponent=1.0),
        cirq.PhasedXPowGate(phase_exponent=-0.5, exponent=1.0)]
    return pi_pulses[idx - 1]

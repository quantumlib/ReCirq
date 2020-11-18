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

"""Functions for generating OTOC circuits."""

from typing import Sequence, Union, Tuple, List, Dict, Set, Any

import cirq
import numpy as np

from recirq.otoc.parallel_xeb import build_xeb_circuits
from recirq.otoc.utils import cz_to_sqrt_iswap

_CORRECTION_FILE = Dict[Tuple[Tuple[int, int], Tuple[int, int]], cirq.Circuit]


def build_otoc_circuits(
        qubits: Sequence[cirq.GridQubit],
        ancilla: cirq.GridQubit,
        cycle: int,
        interaction_sequence: Sequence[Set[Tuple[cirq.GridQubit,
                                                 cirq.GridQubit]]],
        forward_ops: Union[_CORRECTION_FILE, List[_CORRECTION_FILE]],
        reverse_ops: Union[_CORRECTION_FILE, List[_CORRECTION_FILE]],
        butterfly_qubits: Union[cirq.GridQubit, Sequence[cirq.GridQubit]],
        cycles_per_echo: int = None,
        random_seed: int = None,
        sq_gates: np.ndarray = None,
        cz_correction: Dict[Tuple[cirq.GridQubit, cirq.GridQubit],
                            cirq.Circuit] = None,
        use_physical_cz: bool = False,
        light_cone_filter: bool = False,
        cliffords: bool = False,
        z_only: bool = False,
        padding: float = None,
        layer_by_layer_cal: bool = False,
        idle_echo_frequency: int = 1,
        idle_echo_seed: int = None
) -> List[cirq.Circuit]:
    r"""Build experimental circuits for measuring OTOCs.

    A random circuit, $U$, composed of alternating layers of random single-qubit
    gates and fixed two-qubit gates is first created, similar to those used
    in an XEB experiment. The inverse of the random circuit, $U^\dagger$,
    is also created and added after $U$. A perturbation, in the form of Pauli
    operators I, X, Y, or Z on one or more 'butterfly' qubits, is added
    between $U$ and $U^\dagger$. An ancilla qubit interacts with one qubit in
    the system through a CZ gate at the beginning and the end of the quantum
    circuit, and single-qubit readout on the ancilla is performed at the end
    of the quantum circuit.

    Args:
        qubits: The qubits involved in the random circuits $U$ and $U^\dagger$.
        ancilla: The ancilla qubit to measure OTOC with.
        cycle: The number of interleaved single- and two-qubit gate layers in
            $U$ (same in $U^\dagger$).
        interaction_sequence: Qubit pairs that interact in each cycle. If the
            number of cycles is larger than len(interaction_sequence),
            the interacting qubit pairs will repeat themselves every
            len(interaction_sequence) cycles.
        forward_ops: The two-qubit gates assigned to each qubit pair in $U$.
            Each two-qubit gate is represented as a cirq.Circuit. It could be a
            circuit made of a single gate such as cirq.CZ, or a collection of
            gates such as cirq.CZ with multiple cirq.Z gates (which may be
            obtained from e.g. parallel-xeb calibrations). If specified as a
            list (must have a length equal to cycle), each cycle will use gates
            contained in the corresponding element of the list.
        reverse_ops: The two-qubit gates assigned to each qubit pair in
            $U^\dagger$, having the same format as forward_ops.
        butterfly_qubits: The qubits to which a perturbation between $U$ and
            $U^\dagger$ is to be applied.
        cycles_per_echo: How often a spin-echo pulse (a Y gate) is to be
            applied to the ancilla during its idling time. For example,
            if specified to be 2, a Y gate is applied every two cycles.
        random_seed: The seed for the random single-qubit gates. If
            unspecified, no seed will be used.
        sq_gates: Indices for the random single-qubit gates. Dimension should
            be len(qubits) $\times$ cycle. The values should be integers from 0
            to 7 if z_only is False, and floats between -1 and 1 if z_only is
            True.
        cz_correction: Correction to the composite CZ gate, obtained from
            parallel-XEB. If unspecified, the ideal decomposition for CZ into
            sqrt-iSWAP and single-qubit gates will be used.
        use_physical_cz: If True, a direct CZ gate will be used (instead of
            composite gate from sqrt-iSWAP and single-qubit gates).
        light_cone_filter: If True, gates outside the light-cones of the
            butterfly qubit(s) will be removed and replaced with spin-echo
            gates. Gates outside the light-cone of the measurement qubit (
            i.e. qubits[0]) will be completely removed.
        cliffords: If True (and sq_gates is unspecified), Clifford gates
            drawn randomly from +/-X/2 and +/-Y/2 will be used for the
            single-qubit gates.
        z_only: If True, only Z gates (with a random exponent between -1 and
            1) will be used for the random single-qubit gates.
        padding: If specified, a delay with the given value (in nano-seconds)
            will be added between $U$ and $U^\dagger$.
        layer_by_layer_cal: If True, the two-qubit gates in forward_ops and
            reverse_ops are different in each layer (i.e. forward_ops and
            reverse_ops are specified as List[_CORRECTION_FILE]).
        idle_echo_frequency: How often spin-echo pulses (random +/-X or +/-Y
            gates) are applied to qubits outside the light-cones of the
            butterfly qubit(s). Has no effect if light_cone_filter is False.
        idle_echo_seed: The seed for the random spin-echo gates for qubits
            outside the light-cones of the butterfly qubit(s). If
            unspecified, no seed will be used. Has no effect if
            light_cone_filter is False.

    Returns:
        A list of 16 cirq.Circuits. The first 4 circuits are OTOC circuits
        with the butterfly operators being I (i.e. no perturbation between U
        and $U^\dagger$). Circuits 4 to 8, 8 to 12 and 12 to 16 correspond to
        the butterfly operators being X, Y and Z, respectively. For each
        case, the OTOC measurement result is $(p_0 - p_1 - p_2 + p_3) / 2$,
        where $p_i$ is the excited state probability of the $i^\text{th}$
        circuit for the corresponding butterfly operator.
    """
    if cliffords and z_only:
        raise ValueError('cliffords and z_only cannot both be True')

    num_ops = len(interaction_sequence)
    num_qubits = len(qubits)
    multi_b = True if np.ndim(butterfly_qubits) == 1 else False

    if random_seed is not None:
        np.random.seed(random_seed)

    if sq_gates is not None:
        rand_nums = sq_gates[:, 0:cycle]
    else:
        if cliffords:
            rand_nums = np.random.choice(4, (num_qubits, cycle)) * 2
        elif z_only:
            rand_nums = np.random.uniform(-1, 1, (num_qubits, cycle))
        else:
            rand_nums = np.random.choice(8, (num_qubits, cycle))
    rand_nums_rev = _reverse_sq_indices(rand_nums, z_only=z_only)

    if not light_cone_filter:
        moments_for = _compile_moments(interaction_sequence, forward_ops,
                                       layer_by_layer_cal=layer_by_layer_cal)
        moments_back_0 = _compile_moments(interaction_sequence, reverse_ops,
                                          layer_by_layer_cal=layer_by_layer_cal)
        end_layer = (cycle - 1) % num_ops
        moments_back = list(moments_back_0[end_layer::-1])
        moments_back.extend(list(moments_back_0[-1:end_layer:-1]))

        circuits_forward, _ = build_xeb_circuits(
            qubits, [cycle], moments_for, sq_rand_nums=rand_nums,
            z_only=z_only, ancilla=ancilla, cycles_per_echo=cycles_per_echo)
        circuits_backward, _ = build_xeb_circuits(
            qubits, [cycle], moments_back, sq_rand_nums=rand_nums_rev,
            z_only=False, ancilla=ancilla, cycles_per_echo=cycles_per_echo,
            reverse=True)

    else:
        light_cone_u = _extract_light_cone(
            butterfly_qubits, cycle, False, True, interaction_sequence,
            multiple_butterflies=multi_b)
        light_cone_u_dagger = _extract_light_cone(
            butterfly_qubits, cycle, True, False, interaction_sequence,
            multiple_butterflies=multi_b)
        light_cone_meas = _extract_light_cone(
            qubits[0], cycle, True, True, interaction_sequence,
            multiple_butterflies=False)

        full_cone = [*light_cone_u, *light_cone_u_dagger]
        idle_echo_indices = np.zeros((num_qubits, cycle * 2), dtype=int)

        if idle_echo_seed is not None:
            np.random.seed(idle_echo_seed)

        for idx, qubit in enumerate(qubits):
            idle_cycles = []
            for i, cone in enumerate(full_cone):
                if qubit not in cone:
                    idle_cycles.append(i)
            echo_locs = list(range(idle_echo_frequency - 1, len(idle_cycles),
                                   idle_echo_frequency))
            if len(echo_locs) % 2 == 1:
                echo_locs.remove(echo_locs[-1])
            if idle_echo_seed is None:
                rand_echoes = np.ones(int(len(echo_locs) / 2), dtype=int) + 1
            else:
                rand_echoes = np.random.choice(4, int(len(echo_locs) / 2)) + 1
            for k, echo_idx in enumerate(rand_echoes):
                idle_echo_indices[idx, idle_cycles[echo_locs[2 * k]]] = echo_idx
                idle_echo_indices[idx, idle_cycles[
                    echo_locs[2 * k + 1]]] = (echo_idx + 1) % 4 + 1

        echo_indices_for = idle_echo_indices[:, 0:cycle]
        echo_indices_back = idle_echo_indices[:, cycle:(2 * cycle)]

        seq_for, seq_back = _compile_interactions(
            qubits[0], butterfly_qubits, cycle, interaction_sequence)
        moments_for = _compile_moments(seq_for, forward_ops,
                                       layer_by_layer_cal=layer_by_layer_cal)
        moments_back = _compile_moments(seq_back, reverse_ops,
                                        layer_by_layer_cal=layer_by_layer_cal)

        circuits_forward, _ = build_xeb_circuits(
            qubits, [cycle], moments_for, sq_rand_nums=rand_nums,
            z_only=z_only, ancilla=ancilla, cycles_per_echo=cycles_per_echo,
            light_cones=[light_cone_u], echo_indices=echo_indices_for)
        circuits_backward, _ = build_xeb_circuits(
            qubits, [cycle], moments_back, sq_rand_nums=rand_nums_rev,
            z_only=z_only, ancilla=ancilla, cycles_per_echo=cycles_per_echo,
            reverse=True, light_cones=[light_cone_u_dagger, light_cone_meas],
            echo_indices=echo_indices_back)

    butterfly_ops = [None, cirq.X, cirq.Y, cirq.Z]
    prep_phases = [0.0, 1.0]
    meas_phases = [0.0, 1.0]
    otoc_circuits = []

    for b_ops in butterfly_ops:
        for p_prep in prep_phases:
            for p_meas in meas_phases:
                init_ops = [cirq.PhasedXPowGate(
                    phase_exponent=p_prep, exponent=0.5)(ancilla),
                            cirq.Y(qubits[0]) ** 0.5]
                circuit_full = cirq.Circuit(init_ops)

                if use_physical_cz:
                    circuit_full.append(cirq.CZ(ancilla, qubits[0]))
                    additional_layer = []
                else:
                    cz_seq = cz_to_sqrt_iswap(ancilla, qubits[0],
                                              corrections=cz_correction)
                    circuit_full.append(cz_seq[:-1])
                    additional_layer = cz_seq[-1:]

                if z_only:
                    for q in qubits[1:]:
                        if (q.row + q.col) % 2 == 0:
                            additional_layer.append(cirq.Y(q))
                else:
                    additional_layer.append(
                        [cirq.Y(q) ** 0.5 for q in qubits[1:]])

                circuit_full.append(additional_layer)
                circuit_full.append(circuits_forward[0])
                if padding is not None:
                    moment_pad = cirq.Moment(
                        [cirq.WaitGate(duration=cirq.Duration(
                            nanos=padding))(q) for q in qubits])
                    circuit_full.append(moment_pad)

                if b_ops is not None:
                    if multi_b is True:
                        b_mom = cirq.Moment(
                            [b_ops(q_b) for q_b in butterfly_qubits])
                        circuit_full.append(b_mom)
                    else:
                        circuit_full.append(
                            b_ops(butterfly_qubits),
                            strategy=cirq.InsertStrategy.NEW)

                circuit_full.append(circuits_backward[0])

                if use_physical_cz:
                    circuit_full.append(cirq.CZ(ancilla, qubits[0]))
                else:
                    circuit_full.append(cz_to_sqrt_iswap(
                        ancilla, qubits[0], corrections=cz_correction))

                circuit_full.append(cirq.PhasedXPowGate(
                    phase_exponent=p_meas, exponent=0.5)(ancilla))
                circuit_full.append(cirq.measure(ancilla, key='z'),
                                    strategy=cirq.InsertStrategy.NEW)
                otoc_circuits.append(circuit_full)
    return otoc_circuits


def replace_gates(
        meas_qubit: Any,
        butterfly_qubits: Union[Any, Sequence[Any]],
        cycle: int,
        qubit_order: List[Any],
        sq_gates: np.ndarray,
        interaction_sequence: Sequence[Set[Tuple[Any, Any]]],
        num_replaced_gates: int,
        rand_seed: int = None,
        to_cifford_only: bool = False,
        rand_seed_replaced_gates: int = None
) -> np.ndarray:
    r"""Replace a select number of single-qubit gatrs with other random gates.

    The replaced gates always reside within the lightcones of both the
    butterfly qubit(s) and the measurement qubit of an OTOC circuit. The new
    gates are non-Clifford gates by default, chosen randomly from $\pi/2$
    rotations around 4 different axes on the equatorial plane of the Bloch
    sphere ($\pi/4$, $3\pi/4$, $5\pi/4$ or $7\pi/4$ radians from the x-axis).

    Args:
        meas_qubit: The measurement qubit used in the OTOC experiment. Can be
            specified in any format (e.g. a cirq.GridQubit object or Tuple[
            int, int]).
        butterfly_qubits: The qubits to which a perturbation between
            random circuits $U$ and $U^\dagger$ is to be applied.
        cycle: The number of interleaved single- and two-qubit gate layers in
            $U$ (same in $U^\dagger$).
        qubit_order: The order for all qubits involved in $U$.
        sq_gates: Indices for the random single-qubit gates. Dimension should
            be len(qubit_order) $\times$ cycle. The values should be integers
            from 0 to 7.
        interaction_sequence: Qubit pairs that interact in each cycle. If the
            number of cycles is larger than len(interaction_sequence),
            the interacting qubit pairs will repeat themselves every
            len(interaction_sequence) cycles.
        num_replaced_gates: The total number of single-qubit gates to replace.
        rand_seed: The random seed for the locations of the replaced gates.
            If unspecified, no random seed will be used.
        to_cifford_only: If True, the replaced gates will be drawn randomly
            from +/-X/2 and +/-Y/2.
        rand_seed_replaced_gates: The random seed for the gate replacements.

    Returns:
        An np.ndarray with dimensions len(qubit_order) $\times$ cycle.
    """
    if np.ndim(meas_qubit) != np.ndim(butterfly_qubits):
        multiple_butterflies = True
    else:
        multiple_butterflies = False

    butterfly_cone = _extract_light_cone(
        butterfly_qubits, cycle, False, True, interaction_sequence,
        multiple_butterflies=multiple_butterflies)
    measurement_cone = _extract_light_cone(
        meas_qubit, cycle, False, False, interaction_sequence,
        multiple_butterflies=False)
    q_sets = [b.intersection(m) for (b, m) in
              zip(butterfly_cone, measurement_cone)]

    pairs_set = []
    for i in range(len(interaction_sequence)):
        pairs_i = set({})
        for (q0, q1) in interaction_sequence[i]:
            pairs_i.add(q0)
            pairs_i.add(q1)
        pairs_set.append(pairs_i)

    locs = []
    for j, q_set in enumerate(q_sets):
        for q in sorted(list(q_set)):
            if q in pairs_set[j % len(pairs_set)]:
                locs.append((qubit_order.index(q), j))

    if rand_seed is not None:
        np.random.seed(rand_seed)
    np.random.shuffle(locs)
    random_locs = locs[0:num_replaced_gates]

    if rand_seed_replaced_gates is not None:
        np.random.seed(rand_seed_replaced_gates)

    add_num = 0 if to_cifford_only else 1
    random_indices = np.random.choice(4, num_replaced_gates) * 2 + add_num

    new_sq_gates = sq_gates.copy()
    for k, r in enumerate(random_locs):
        new_sq_gates[r[0], r[1]] = random_indices[k]

    return new_sq_gates


def _compile_moments(
        ops_steps: Sequence[Set[Tuple[cirq.GridQubit, cirq.GridQubit]]],
        ops_dict: Union[_CORRECTION_FILE, List[_CORRECTION_FILE]],
        layer_by_layer_cal: bool = False
) -> List[Union[cirq.Moment, Sequence[cirq.Moment]]]:
    moment_list = []
    num_slices = 3
    if layer_by_layer_cal:
        for file in ops_dict:
            if len(file) > 0:
                num_slices = len(list(file.values())[0])
    else:
        num_slices = len(list(ops_dict.values())[0])

    for i, step in enumerate(ops_steps):
        if len(step) == 0:
            moment_list.append(cirq.Moment([]))
        else:
            ops_list = [[] for _ in range(num_slices)]
            for pair in step:
                pair_locs = ((pair[0].row, pair[0].col),
                             (pair[1].row, pair[1].col))
                for s in range(num_slices):
                    if layer_by_layer_cal:
                        ops_list[s].extend(
                            list(ops_dict[i][pair_locs][s].operations))
                    else:
                        ops_list[s].extend(
                            list(ops_dict[pair_locs][s].operations))
            if num_slices == 1:
                moment_list.append(cirq.Moment(ops_list[0]))
            else:
                moment_list.append([cirq.Moment(m) for m in ops_list])
    return moment_list


def _compile_interactions(first_qubit: Any,
                          butterfly_qubit: Union[Any, Sequence[Any]],
                          cycle: int,
                          interaction_sequence: Sequence[Set[Tuple[Any, Any]]]
                          ) -> Tuple[List[Set[Tuple[Any, Any]]],
                                     List[Set[Tuple[Any, Any]]]]:
    num_configs = len(interaction_sequence)

    if np.ndim(first_qubit) != np.ndim(butterfly_qubit):
        multiple_butterflies = True
    else:
        multiple_butterflies = False

    forward_cone = _extract_light_cone(
        butterfly_qubit, cycle, False, True, interaction_sequence,
        multiple_butterflies=multiple_butterflies)
    forward_seq = []
    for i, cone in enumerate(forward_cone):
        int_list = []
        for pair in interaction_sequence[i % num_configs]:
            if pair[0] in cone or pair[1] in cone:
                int_list.append(pair)
        forward_seq.append(set(int_list))

    reverse_cone_0 = _extract_light_cone(
        first_qubit, cycle, True, True, interaction_sequence,
        multiple_butterflies=False)
    reverse_seq = []
    reverse_seq_0 = list(reversed(forward_seq))
    for i, int_list in enumerate(reverse_seq_0):
        new_set = set({})
        for pair in int_list:
            if (pair[0] in reverse_cone_0[i]) and (
                    pair[1] in reverse_cone_0[i]):
                new_set.add(pair)
        reverse_seq.append(new_set)

    return forward_seq, reverse_seq


def _extract_light_cone(butterfly_qubits: Union[Any, Sequence[Any]],
                        cycle: int, reverse: bool,
                        from_right: bool,
                        interaction_layers: Sequence[Set[Tuple[Any, Any]]],
                        multiple_butterflies: bool = False
                        ) -> List[Set[Any]]:
    num_configs = len(interaction_layers)
    int_seq = [interaction_layers[c % num_configs] for c in range(cycle)]
    if reverse ^ from_right:
        int_seq.reverse()

    if multiple_butterflies is True:
        current_set = {*butterfly_qubits}
    else:
        current_set = {butterfly_qubits}

    qubit_set_list = []
    for layer in int_seq:
        current_set = _add_layer(current_set, layer)
        qubit_set_list.append(current_set)

    if from_right:
        qubit_set_list.reverse()

    return qubit_set_list


def _add_layer(set_0: Set[Any], sets_1: Set[Tuple[Any, Any]]
               ) -> Set[Any]:
    new_set = set_0.copy()
    for set_1 in sets_1:
        q0, q1 = set_1
        if (q0 in set_0) or (q1 in set_0):
            new_set.update({q0, q1})
    return new_set


def _reverse_sq_indices(indices: np.ndarray, z_only: bool = False
                        ) -> np.ndarray:
    if z_only:
        rev_indices = -indices[:, ::-1]
    else:
        rev_indices = indices.copy()
        _, num_layers = indices.shape
        for (i, j), v in np.ndenumerate(indices):
            rev_indices[i, num_layers - j - 1] = v + 4 if v < 4 else v - 4
    return rev_indices

# Copyright 2024 Google
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
from math import gcd
from typing import Dict, Iterable, List, Sequence, Type, cast

import cirq
import numpy as np

from recirq.qcqmc.config import (
    DO_INVERSE_SIMULATION_QUBIT_NUMBER_CUTOFF,
    SINGLE_PRECISION_DEFAULT,
)
from recirq.qcqmc.for_refactor import (
    is_expected_elementary_cirq_op,
    reorder_qubit_wavefunction,
)


def reconstruct_wavefunctions_from_samples(
    *,
    raw_samples: np.ndarray,
    factorized_cliffords: Sequence[Sequence[cirq.Circuit]],
    qubit_partition: Sequence[Sequence[cirq.Qid]],
    valid_configurations: Sequence[Sequence[bool]],
    k_to_calculate: Sequence[int],
    qubits_linearly_connected: Sequence[cirq.Qid],
    qubits_jordan_wigner_ordered: Sequence[cirq.Qid],
    simulate_single_precision: bool = SINGLE_PRECISION_DEFAULT,
) -> Dict[str, np.ndarray]:
    """Reconstructs the wavefunction from the given samples.

    This method is responsible for turning the samples into a classical shadow
    and evaluating the matrix elements required to reconstruct a wavefunction
    that is a linear combination of the states specified by
    valid_configurations.

    Args:
        raw_samples: The measured bitstrings of shape (n_cliffords,
            n_samples_per_clifford, n_qubits)
        factorized_cliffords: A list of lists of factorized random clifford circuits.
            Each list of clifford circuits factorized_cliffords should
            act on the Qids in the appropiate qubit_partition.
        qubit_partition: A list of lists of Qids to which the corresponding
            factorized clifford circuits are applied to. Should be the same
            length as factorized_cliffords.
        valid_configurations: We only want to reconstruct the amplitudes of the
            wavefunction on computational basis states with the right symmetry.
            Each entry in the list is a list of bools which encode  valid
            computational basis states a specific symmetry.
        k_to_calculate: Specifying k_to_calculate lets you use the same samples
            to estimate the amplitudes with various choices of k for the median of
            means estimator. (1,) corresponds to just taking the mean of all the
            samples. (3,) corresponds to dividing them into three groups and taking
            the median of the mean of each of these groups. (1, 3) uses the samples
            in both ways, which is useful for comparison.
        qubits_linearly_connected: The qubits in the order that the circuit was
            ran in / samples were taken in.
        qubits_jordan_wigner_ordered: The qubits in the order that is needed to
            compare to the original trial wavefunction.
        simulate_single_precision: Single or double precision simulation.
    """

    n_cliffords, _, n_qubits = raw_samples.shape
    assert n_cliffords == len(factorized_cliffords)

    # Let's collect all of the qubits in their original order.
    joint_qubits = []
    for part in qubit_partition:
        joint_qubits += list(part)
    joint_qubits = list(sorted(joint_qubits))
    assert len(qubits_linearly_connected) == len(frozenset(joint_qubits))
    n_qubits = len(qubits_linearly_connected)
    qubit_to_index = {qubit: i for i, qubit in enumerate(qubits_linearly_connected)}

    lcm_of_ks = get_lcm(k_to_calculate)
    amplitude_sets = np.zeros((lcm_of_ks, len(valid_configurations)), np.complex128)
    counts = [0 for _ in range(lcm_of_ks)]

    # Now we apply the inverse circuits to the samples.
    for i in range(n_cliffords):
        inverse_cliffords = [cirq.inverse(cliff) for cliff in factorized_cliffords[i]]

        assert len(inverse_cliffords) == len(qubit_partition)
        for part, inverse in zip(qubit_partition, inverse_cliffords):
            assert frozenset(part) == inverse.all_qubits()

        amplitudes = get_amplitudes_from_samples(
            inverse_cliffords=inverse_cliffords,
            qubit_partition=qubit_partition,
            raw_samples_subset=raw_samples[i, :, :],
            valid_configurations=valid_configurations,
            qubit_to_index=qubit_to_index,
            simulate_single_precision=simulate_single_precision,
        )

        index = i % lcm_of_ks
        amplitude_sets[index, :] += amplitudes
        counts[index] += 1

    # Now we average and take the medians.
    for i, count in enumerate(counts):
        if count != 0:
            amplitude_sets[i, :] /= count

    wavefunctions_for_various_k = {}
    for k in k_to_calculate:
        k_amplitude_sets = np.zeros((k, len(valid_configurations)), complex)
        counts = [0 for _ in range(k)]
        for i in range(lcm_of_ks):
            index = i % k
            k_amplitude_sets[index, :] += amplitude_sets[i, :]
            counts[index] += 1
        for i, count in enumerate(counts):
            k_amplitude_sets[i, :] /= count

        real_medians = np.median(np.real(k_amplitude_sets), axis=0)
        imag_medians = np.median(np.imag(k_amplitude_sets), axis=0)

        wf = np.zeros((2**n_qubits,), complex)

        for i, config in enumerate(valid_configurations):
            index = cirq.big_endian_bits_to_int(config)
            wf[index] = real_medians[i] + 1j * imag_medians[i]

        wf = reorder_qubit_wavefunction(
            wf=wf,
            qubits_old_order=qubits_linearly_connected,
            qubits_new_order=qubits_jordan_wigner_ordered,
        )

        # We need a string key here for json saving and loading later.
        wavefunctions_for_various_k[str(k)] = wf

    return wavefunctions_for_various_k


def get_amplitudes_from_samples(
    *,
    inverse_cliffords: List[cirq.Circuit],
    qubit_partition: Sequence[Sequence[cirq.Qid]],
    raw_samples_subset: np.ndarray,
    valid_configurations: Sequence[Sequence[bool]],
    qubit_to_index: Dict[cirq.Qid, int],
    simulate_single_precision: bool,
) -> np.ndarray:
    """A helper function to reconstruct the wavefunction amplitudes from samples.

    raw_samples_subset is a subset of the samples taken in the experiment,
    all taken from repeating a single circuit (whose Clifford is inverted by the
    collection of Cliffords in inverse_cliffords).

    This will use the correct method depending on the largest qubit partition.
    """
    max_part_len = max(len(part) for part in qubit_partition)

    if max_part_len > DO_INVERSE_SIMULATION_QUBIT_NUMBER_CUTOFF:
        func = get_amplitudes_from_samples_via_clifford_simulation

    else:
        func = get_amplitudes_from_samples_via_big_unitary

    return func(
        inverse_cliffords=inverse_cliffords,
        qubit_partition=qubit_partition,
        raw_samples_subset=raw_samples_subset,
        valid_configurations=valid_configurations,
        qubit_to_index=qubit_to_index,
        simulate_single_precision=simulate_single_precision,
    )


def get_amplitudes_from_samples_via_big_unitary(
    *,
    inverse_cliffords: List[cirq.Circuit],
    qubit_partition: Sequence[Sequence[cirq.Qid]],
    raw_samples_subset: np.ndarray,
    valid_configurations: Sequence[Sequence[bool]],
    qubit_to_index: Dict[cirq.Qid, int],
    simulate_single_precision: bool = SINGLE_PRECISION_DEFAULT,
) -> np.ndarray:
    """A helper function to reconstruct the wavefunction amplitudes from samples.

    raw_samples_subset is a subset of the samples taken in the experiment,
    all taken from repeating a single circuit (whose Clifford is inverted by the
    collection of Cliffords in inverse_cliffords).


    for small systems it will be faster to explicitly construct the
    unitary matrix corresponding to the inverse.
    """
    n_samples = raw_samples_subset.shape[0]
    assert len(raw_samples_subset.shape) == 2
    amplitudes = np.zeros((len(valid_configurations),), np.complex128)
    dtype = cast(
        Type[np.complexfloating],
        np.complex64 if simulate_single_precision else np.complex128,
    )

    inverse_unitaries = []
    for part, inverse in zip(qubit_partition, inverse_cliffords):
        qubit_order = cirq.QubitOrder.explicit(part)
        unitary = inverse.unitary(dtype=dtype, qubit_order=qubit_order)
        inverse_unitaries.append(unitary)

    for l, config in enumerate(valid_configurations):
        config_partition = []
        for part in qubit_partition:
            config_part = []
            for qubit in part:
                config_part.append(config[qubit_to_index[qubit]])
            config_partition.append(config_part)

        for j in range(n_samples):
            zero_bitstring_matrix_element = 1.0
            for unitary, part, config_part in zip(
                inverse_unitaries, qubit_partition, config_partition
            ):
                bitstring = [
                    raw_samples_subset[j, qubit_to_index[qubit]] for qubit in part
                ]
                bitstring_val = cirq.big_endian_bits_to_int(bitstring)
                sub_shadow = unitary[:, bitstring_val]

                config_val = cirq.big_endian_bits_to_int(config_part)
                c = np.conj(sub_shadow[0]) * sub_shadow[config_val]
                c *= 2 ** len(part) + 1
                if config_val == 0:
                    # if the matrix element is diagonal we subtract one to account for the
                    # subtraction of the identity matrix.
                    c -= 1

                zero_bitstring_matrix_element *= c

            amplitudes[l] += zero_bitstring_matrix_element * 2

    # we average and return.
    amplitudes /= n_samples

    return amplitudes


def get_amplitudes_from_samples_via_simulation(
    *,
    inverse_cliffords: List[cirq.Circuit],
    qubit_partition: Sequence[Sequence[cirq.Qid]],
    raw_samples_subset: np.ndarray,
    valid_configurations: Sequence[Sequence[bool]],
    qubit_to_index: Dict[cirq.Qid, int],
    simulate_single_precision: bool = SINGLE_PRECISION_DEFAULT,
) -> np.ndarray:
    """A helper function to reconstruct the wavefunctions amplitudes from samples.

    This method simulates the inverse circuits, which is faster for sufficiently
    large circuits.

    raw_samples_subset is a subset of the samples taken in the experiment,
    all taken from repeating a single circuit (whose Clifford is inverted by the
    collection of Cliffords in inverse_cliffords).
    """
    n_samples = raw_samples_subset.shape[0]
    assert len(raw_samples_subset.shape) == 2
    amplitudes = np.zeros((len(valid_configurations),), np.complex128)

    assert simulate_single_precision, "TODO"
    simulator = cirq.Simulator()

    partitioned_results = []
    for part, inverse in zip(qubit_partition, inverse_cliffords):
        initial_states = [
            cirq.big_endian_bits_to_int(
                [raw_samples_subset[j, qubit_to_index[qubit]] for qubit in part]
            )
            for j in range(n_samples)
        ]

        results = [
            simulator.simulate(inverse, qubit_order=part, initial_state=init_state)
            for init_state in initial_states
        ]
        partitioned_results.append(results)

    for j in range(n_samples):
        inverted_wavefunctions = []
        for part, inverse, results in zip(
            qubit_partition, inverse_cliffords, partitioned_results
        ):
            result = results[j]
            inverted_wavefunctions.append(result.state_vector())

        for l, config in enumerate(valid_configurations):
            config_partition = []
            for part in qubit_partition:
                config_part = []
                for qubit in part:
                    config_part.append(config[qubit_to_index[qubit]])
                config_partition.append(config_part)

            zero_bitstring_matrix_element = 1.0
            for sub_shadow, part, config_part in zip(
                inverted_wavefunctions, qubit_partition, config_partition
            ):

                config_val = cirq.big_endian_bits_to_int(config_part)
                c = np.conj(sub_shadow[0]) * sub_shadow[config_val]
                c *= 2 ** len(part) + 1
                if config_val == 0:
                    # if the matrix element is diagonal we subtract one to account for the
                    # subtraction of the identity matrix.
                    c -= 1

                zero_bitstring_matrix_element *= c

            amplitudes[l] += zero_bitstring_matrix_element * 2

    # we average and return.
    amplitudes /= n_samples
    return amplitudes


def get_amplitudes_from_samples_via_clifford_simulation(
    *,
    inverse_cliffords: List[cirq.Circuit],
    qubit_partition: Sequence[Sequence[cirq.Qid]],
    raw_samples_subset: np.ndarray,
    valid_configurations: Sequence[Sequence[bool]],
    qubit_to_index: Dict[cirq.Qid, int],
    simulate_single_precision: bool = SINGLE_PRECISION_DEFAULT,
) -> np.ndarray:
    """A helper function to reconstruct the wavefunctions amplitudes from samples.

    This method simulates the inverse circuits using an efficient Clifford circuit
    simulator, which is faster for sufficiently large circuits.

    raw_samples_subset is a subset of the samples taken in the experiment,
    all taken from repeating a single circuit (whose Clifford is inverted by the
    collection of Cliffords in inverse_cliffords).
    """
    n_samples = raw_samples_subset.shape[0]
    assert len(raw_samples_subset.shape) == 2
    amplitudes = np.zeros((len(valid_configurations),), np.complex128)

    assert simulate_single_precision, "TODO"
    simulator = cirq.CliffordSimulator()

    decomposed_inverse_cliffords = []

    for inverse_clifford in inverse_cliffords:
        decomposed_inverse_clifford = cirq.decompose(
            inverse_clifford, keep=is_expected_elementary_cirq_op
        )

        decomposed_inverse_cliffords.append(cirq.Circuit(decomposed_inverse_clifford))

    partitioned_results = []
    for part, inverse in zip(qubit_partition, decomposed_inverse_cliffords):
        initial_states = [
            cirq.big_endian_bits_to_int(
                [raw_samples_subset[j, qubit_to_index[qubit]] for qubit in part]
            )
            for j in range(n_samples)
        ]

        results = [
            simulator.simulate(inverse, qubit_order=part, initial_state=init_state)
            for init_state in initial_states
        ]
        partitioned_results.append(results)

    for j in range(n_samples):
        inverted_wavefunctions = []
        for part, inverse, results in zip(
            qubit_partition, decomposed_inverse_cliffords, partitioned_results
        ):
            result = results[j]
            inverted_wavefunctions.append(result.final_state.state_vector())

        for l, config in enumerate(valid_configurations):
            config_partition = []
            for part in qubit_partition:
                config_part = []
                for qubit in part:
                    config_part.append(config[qubit_to_index[qubit]])
                config_partition.append(config_part)

            zero_bitstring_matrix_element = 1.0
            for sub_shadow, part, config_part in zip(
                inverted_wavefunctions, qubit_partition, config_partition
            ):

                config_val = cirq.big_endian_bits_to_int(config_part)
                c = np.conj(sub_shadow[0]) * sub_shadow[config_val]
                c *= 2 ** len(part) + 1
                if config_val == 0:
                    # if the matrix element is diagonal we subtract one to account for the
                    # subtraction of the identity matrix.
                    c -= 1

                zero_bitstring_matrix_element *= c

            amplitudes[l] += zero_bitstring_matrix_element * 2

    # we average and return.
    amplitudes /= n_samples
    return amplitudes


def get_lcm(ks: Iterable[int]) -> int:
    """Gets the least common multiple of a Tuple of integers."""
    lcm = 1

    for k in ks:
        lcm = lcm * k // gcd(lcm, k)

    return lcm

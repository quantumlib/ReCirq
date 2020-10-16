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
"""Circuit generator that prepares a single-fermion initial states."""

import math
from typing import List, Optional, Tuple, Sequence, Iterable

import cirq
import numpy as np


def create_one_particle_circuit(
        qubits: List[cirq.Qid],
        amplitudes: np.array,
        x_index: Optional[int] = None
) -> cirq.Circuit:
    """Construct an arbitrary single-particle fermionic state.

    This procedure constructs circuit of a lower depth compared to the more
    generic openfermion.slater_determinant_preparation_circuit. It allows to
    initializes the qubit in the middle as starting excitation and distributes
    the amplitudes to the left and right simultaneously, which cuts the depth
    by about two.

    Args:
        qubits: List of qubits.
        amplitudes: The desired coefficient for each fermion.
        x_index: Index of the initial excitation to propagate from.

    Returns:
        Circuit that realizes the desired distribution.
    """

    sites_count = len(qubits)

    if sites_count != len(amplitudes):
        raise ValueError('Length of amplitudes and qubits array must be equal')

    x_index, rotations, phases = _create_one_particle_moments(
        amplitudes, x_index)

    # Prepare initial state.
    circuit = cirq.Circuit([cirq.X.on(qubits[x_index])])

    # Construct the circuit from the rotations.
    for moment in rotations:
        circuit += cirq.Circuit(_create_moment_operations(qubits, moment))

    # Fix the desired phases.
    circuit += cirq.Circuit(
        [cirq.Z(qubits[i]) ** phases[i] for i in range(sites_count)])

    return circuit


def _create_one_particle_moments(
        amplitudes: np.array,
        x_index: Optional[int] = None,
        offset: int = 0
) -> Tuple[int, List[Tuple[Tuple[int, int, float], ...]], np.ndarray]:

    if not np.isclose(np.linalg.norm(amplitudes), 1.0):
        raise ValueError('Amplitudes array must be normalized')

    sites_count = len(amplitudes)
    phases = np.angle(amplitudes) / np.pi

    if x_index is None:
        x_index = (sites_count - 1) // 2

    if sites_count == 1:
        return x_index, [], phases

    left_count = x_index
    right_count = sites_count - x_index - 1

    if left_count > right_count:
        right_rotations, previous = _create_rotations(
            list(reversed(amplitudes[x_index:])),
            shift=x_index,
            offset=offset,
            reverse=True)
        left_rotations, _ = _create_rotations(
            list(amplitudes[:x_index]) + [previous],
            shift=0,
            offset=offset,
            reverse=False)
        rotations = _merge_rotations(left_rotations, right_rotations)
    else:
        left_rotations, previous = _create_rotations(
            amplitudes[:x_index + 1],
            shift=0,
            offset=offset,
            reverse=False)
        right_rotations, _ = _create_rotations(
            list(reversed(amplitudes[x_index + 1:])) + [previous],
            shift=x_index,
            offset=offset,
            reverse=True)
        rotations = _merge_rotations(right_rotations, left_rotations)

    return x_index, rotations, phases


def _create_rotations(amplitudes: Sequence[complex],
                      shift: int,
                      offset: int,
                      reverse: bool
                      ) -> Tuple[List[Tuple[int, int, float]], float]:
    rotations = []
    size = len(amplitudes)
    previous = np.abs(amplitudes[0])
    for i in range(1, size):
        theta, previous = _left_givens_params(
            previous, np.abs(amplitudes[i]))
        if reverse:
            rotations.append(
                (size - i + shift - 1 + offset,
                 size - i + shift + offset,
                 -theta))
        else:
            rotations.append(
                (i + shift - 1 + offset,
                 i + shift + offset,
                 theta))
    return list(reversed(rotations)), previous


def _left_givens_params(left, right):
    theta = math.atan2(left, right)
    return theta, left / math.sin(theta)


def _merge_rotations(first: List[Tuple[int, int, float]],
                     second: List[Tuple[int, int, float]]
                     ) -> List[Tuple[Tuple[int, int, float], ...]]:
    rotations = [(first[0],)] + list(zip(first[1:], second))
    count = len(rotations)
    rotations += [(r,) for r in first[count:]]
    rotations += [(r,) for r in second[count - 1:]]
    return rotations


def _create_moment_operations(
        qubits: List[cirq.Qid],
        moment: Iterable[Tuple[int, int, float]]) -> List[cirq.Operation]:
    return [cirq.givens(theta).on(qubits[a], qubits[b])
            for a, b, theta in moment]
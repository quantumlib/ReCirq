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

from typing import List

import cirq
import numpy as np
import pytest

from recirq.fermi_hubbard.fermionic_circuits import create_one_particle_circuit


def _single_fermionic_modes_state(amplitudes: List[complex]) -> np.ndarray:
    """Prepares state which is a superposition of single Fermionic modes.

    Args:
        amplitudes: List of amplitudes to be assigned for a state representing
            a Fermionic mode.

    Return:
        State vector which is a superposition of single fermionic modes under
        JWT representation, each with appropriate amplitude assigned.
    """
    n = len(amplitudes)
    state = np.zeros(1 << n, dtype=complex)
    for m in range(len(amplitudes)):
        state[1 << (n - 1 - m)] = amplitudes[m]
    return state / np.linalg.norm(state)


@pytest.mark.parametrize(
    'shape',
    [
        [1.0],
        [1.0j],
        [1.0, 1.0],
        [1.0, 0.5],
        [1.0, 0.5 + 0.5j],
        [0.5 - 0.5j, 0.5 + 0.5j],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
)
def test_create_one_particle_circuit(shape):
    amplitudes = shape / np.linalg.norm(shape)
    qubits = cirq.LineQubit.range(len(amplitudes))
    circuit = create_one_particle_circuit(qubits, amplitudes)
    assert np.allclose(
        circuit.final_state_vector(),
        _single_fermionic_modes_state(amplitudes))


@pytest.mark.parametrize(
    'shape',
    [
        [1.0],
        [1.0j],
        [1.0, 1.0],
        [1.0, 0.5],
        [1.0, 0.5 + 0.5j],
        [0.5 - 0.5j, 0.5 + 0.5j],
        [1, 2, 3],
        [1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 7],
        [1, 2, 3, 4, 5, 6, 7, 8],
        [1, 1, 1, 1, 1, 1, 1, 1]
    ]
)
def test_create_one_particle_circuit_with_x_index(shape):
    for x_index in range(len(shape)):
        amplitudes = shape / np.linalg.norm(shape)
        qubits = cirq.LineQubit.range(len(amplitudes))
        circuit = create_one_particle_circuit(qubits, amplitudes, x_index)
        assert np.allclose(
            circuit.final_state_vector(),
            _single_fermionic_modes_state(amplitudes))

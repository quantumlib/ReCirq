# Copyright 2023 Google
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

# Test that decomposed gates act the same as the given unitary
import cirq
import numpy as np

from recirq.seniority_zero.circuits_simplified.gates import (
    GSGate,
    ZImXYRotationGate,
    ZIpXYRotationGate,
    ZZRotationGate,
)


def test_GSGate():
    angle = 0.5431
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    gate = GSGate(angle)

    decomposition = cirq.Circuit(gate._decompose_([q0, q1]))
    unitary = gate._unitary_()
    unitary_test = cirq.unitary(decomposition)
    assert np.abs(np.abs(np.trace(unitary.conj().T @ unitary_test)) - 4) < 1e-3


def test_ZZRotationGate():
    angle = 0.5431
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    gate = ZZRotationGate(angle)

    decomposition = cirq.Circuit(gate._decompose_([q0, q1]))
    unitary = gate._unitary_()
    unitary_test = cirq.unitary(decomposition)
    assert np.abs(np.abs(np.trace(unitary.conj().T @ unitary_test)) - 4) < 1e-3


def test_ZImXYRotationGate():
    angle = 0.5431
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    gate = ZImXYRotationGate(angle)

    decomposition = cirq.Circuit(gate._decompose_([q0, q1]))
    unitary = gate._unitary_()
    unitary_test = cirq.unitary(decomposition)
    print(unitary.round(3))
    print(unitary_test.round(3))
    assert np.abs(np.abs(np.trace(unitary.conj().T @ unitary_test)) - 4) < 1e-3


def test_ZIpXYRotationGate():
    angle = 0.5431
    q0 = cirq.GridQubit(0, 0)
    q1 = cirq.GridQubit(0, 1)
    gate = ZIpXYRotationGate(angle)

    decomposition = cirq.Circuit(gate._decompose_([q0, q1]))
    unitary = gate._unitary_()
    unitary_test = cirq.unitary(decomposition)
    assert np.abs(np.abs(np.trace(unitary.conj().T @ unitary_test)) - 4) < 1e-3

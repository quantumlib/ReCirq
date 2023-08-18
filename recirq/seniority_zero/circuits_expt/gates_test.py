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

"""Validate that the gates are correct for this experiment

we will use the Givens rotation gate from Hartree-Fock experiment
as a base
"""
import cirq
import numpy as np

from recirq.seniority_zero.circuits_expt.gates import (
    cnot_from_cz,
    cnot_from_sqrt_swap,
    gsgate_from_cz,
    gsgate_from_sqrt_swap,
)


def givens_gate(a, b, theta) -> cirq.Circuit:
    """Implements the givens rotation with sqrt(iswap).
    The inverse(sqrt(iswap)) is made with z before and after"""
    givens_gate = cirq.Circuit(
        [
            cirq.ISWAP.on(a, b) ** 0.5,
            cirq.rz(-theta + np.pi).on(a),
            cirq.rz(theta).on(b),
            cirq.ISWAP.on(a, b) ** 0.5,
            cirq.rz(np.pi).on(a),
        ]
    )
    return givens_gate


def test_givens_gate():
    thetas = np.linspace(0, 2 * np.pi, 100)
    qubits = cirq.LineQubit.range(2)
    for tt in thetas:
        u_test = cirq.unitary(givens_gate(qubits[0], qubits[1], tt))
        # set global phase correctly such that [0, 0] and [-1, -1] element
        # have +1 phase
        u_test *= np.sign(u_test[0, 0])
        assert np.isclose(u_test[1, 2], -np.sin(tt))
        assert np.isclose(u_test[1, 1], np.cos(tt))


def test_gsgate_fromcz():
    np.set_printoptions(linewidth=500)
    thetas = np.linspace(0, 2 * np.pi, 100)
    qubits = cirq.LineQubit.range(2)
    for tt in thetas:
        u_test = cirq.unitary(gsgate_from_cz(qubits[0], qubits[1], tt))
        u_true = cirq.unitary(
            cirq.Circuit(
                [givens_gate(qubits[0], qubits[1], tt), cirq.SWAP.on(qubits[0], qubits[1])]
            )
        )
        # calculate overlap of two unitaries
        # should be dimension of Hilbert space
        assert np.isclose(abs(np.trace(u_test.conj().T @ u_true)), 4)


def test_gsgate_from_sqrt_iswap():
    np.set_printoptions(linewidth=500)
    thetas = np.linspace(0, 2 * np.pi, 100)
    qubits = cirq.LineQubit.range(2)
    for tt in thetas:
        u_test = cirq.unitary(gsgate_from_sqrt_swap(qubits[0], qubits[1], tt))
        u_true = cirq.unitary(
            cirq.Circuit(
                [givens_gate(qubits[0], qubits[1], tt), cirq.SWAP.on(qubits[0], qubits[1])]
            )
        )
        # calculate overlap of two unitaries
        # should be dimension of Hilbert space
        assert np.isclose(abs(np.trace(u_test.conj().T @ u_true)), 4)


def test_cnot_from_cz():
    qubits = cirq.LineQubit.range(2)
    test_unitary = cirq.unitary(cnot_from_cz(qubits[0], qubits[1]))
    true_unitary = cirq.unitary(cirq.Circuit([cirq.CNOT.on(qubits[0], qubits[1])]))
    assert np.isclose(abs(np.trace(test_unitary.conj().T @ true_unitary)), 4)


def test_cnot_from_sqrt_iswap():
    qubits = cirq.LineQubit.range(2)
    test_unitary = cirq.unitary(cnot_from_sqrt_swap(qubits[0], qubits[1]))
    true_unitary = cirq.unitary(cirq.Circuit([cirq.CNOT.on(qubits[0], qubits[1])]))
    assert np.isclose(abs(np.trace(test_unitary.conj().T @ true_unitary)), 4)

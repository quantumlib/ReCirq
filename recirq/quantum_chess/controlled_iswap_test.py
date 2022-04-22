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
import cirq
import recirq.quantum_chess.controlled_iswap as controlled_iswap


def test_controlled_iswap():
    qubits = cirq.LineQubit.range(3)
    result = controlled_iswap.controlled_iswap(qubits[0], qubits[1], qubits[2])
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(result),
        cirq.unitary(
            cirq.Circuit(cirq.ISWAP(qubits[0], qubits[1]).controlled_by(qubits[2]))
        ),
        atol=1e-8,
    )


def test_controlled_inv_iswap():
    qubits = cirq.LineQubit.range(3)
    result = controlled_iswap.controlled_iswap(
        qubits[0], qubits[1], qubits[2], inverse=True
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(result),
        cirq.unitary(
            cirq.Circuit(
                (cirq.ISWAP(qubits[0], qubits[1]) ** -1).controlled_by(qubits[2])
            )
        ),
        atol=1e-8,
    )


def test_controlled_sqrt_iswap():
    qubits = cirq.LineQubit.range(3)
    result = controlled_iswap.controlled_sqrt_iswap(qubits[0], qubits[1], qubits[2])
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(result),
        cirq.unitary(
            cirq.Circuit(
                (cirq.ISWAP(qubits[0], qubits[1]) ** 0.5).controlled_by(qubits[2])
            )
        ),
        atol=1e-6,
    )


def test_controlled_inv_sqrt_iswap():
    qubits = cirq.LineQubit.range(3)
    result = controlled_iswap.controlled_inv_sqrt_iswap(qubits[0], qubits[1], qubits[2])
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(result),
        cirq.unitary(
            cirq.Circuit(
                (cirq.ISWAP(qubits[0], qubits[1]) ** -0.5).controlled_by(qubits[2])
            )
        ),
        atol=1e-6,
    )

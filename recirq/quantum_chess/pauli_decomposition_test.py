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
import pytest
import cirq
import numpy as np
import re

from recirq.quantum_chess.pauli_decomposition import pauli_decomposition


def test_pauli_decomposition_wrong_inputs():
    a1 = cirq.NamedQubit("a1")
    a2 = cirq.NamedQubit("a2")
    H_not_2d = [[[0.5 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.0j]]]
    H_not_square = [[0.5 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j]]
    H_good = [[0.5 + 0.0j, 0.0 + 0.5j], [0.0 - 0.5j, 0.5 + 0.0j]]
    with pytest.raises(
        ValueError, match="pauli_decomposition expects a 2-d square matrix."
    ):
        pauli_decomposition(H_not_2d, [a1])

    with pytest.raises(
        ValueError, match="pauli_decomposition expects a 2-d square matrix."
    ):
        pauli_decomposition(H_not_square, [a1])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "pauli_decomposition: Expect that size_of_matrix==pow(2, number_of_qubits). In your case 2!=pow(2, 2)."
        ),
    ):
        pauli_decomposition(H_good, [a1, a2])


@pytest.mark.parametrize(
    "measurement",
    (
        np.random.rand(2, 2),
        np.zeros((2, 2)),
        np.identity(2),
    ),
)
def test_pauli_decomposition_1_qubit(measurement):
    a1 = cirq.NamedQubit("a1")
    decomp = pauli_decomposition(measurement, [a1])
    assert np.allclose(measurement, decomp.matrix([a1]))


@pytest.mark.parametrize(
    "measurement",
    (
        np.random.rand(4, 4),
        np.zeros((4, 4)),
        np.identity(4),
    ),
)
def test_pauli_decomposition_2_qubit(measurement):
    a1 = cirq.NamedQubit("a1")
    a2 = cirq.NamedQubit("a2")
    for qubits in [[a1, a2], [a2, a1]]:
        decomp = pauli_decomposition(measurement, qubits)
        assert np.allclose(measurement, decomp.matrix(qubits))


@pytest.mark.parametrize(
    "measurement",
    (
        np.random.rand(8, 8),
        np.zeros((8, 8)),
        np.identity(8),
    ),
)
def test_pauli_decomposition_3_qubit(measurement):
    a1 = cirq.NamedQubit("a1")
    a2 = cirq.NamedQubit("a2")
    a3 = cirq.NamedQubit("a3")
    for qubits in [[a1, a2, a3], [a3, a1, a2], [a2, a3, a1]]:
        decomp = pauli_decomposition(measurement, qubits)
        assert np.allclose(measurement, decomp.matrix(qubits))


@pytest.mark.parametrize(
    "measurement",
    (
        np.random.rand(16, 16),
        np.zeros((16, 16)),
        np.identity(16),
    ),
)
def test_pauli_decomposition_4_qubit(measurement):
    a1 = cirq.NamedQubit("a1")
    a2 = cirq.NamedQubit("a2")
    a3 = cirq.NamedQubit("a3")
    b1 = cirq.NamedQubit("b1")
    for qubits in [[a1, a2, a3, b1], [a2, b1, a1, a3], [b1, a3, a2, a1]]:
        decomp = pauli_decomposition(measurement, qubits)
        assert np.allclose(measurement, decomp.matrix(qubits))


@pytest.mark.parametrize(
    "vector, expected_str",
    (
        (
            [0.0, 1.0, 1.0, 0.0],
            "0.250*I+0.250*X(a1)*X(a2)+0.250*Y(a1)*Y(a2)-0.250*Z(a1)*Z(a2)",
        ),
        (
            [0.0, 1.0, -1.0, 0.0],
            "0.250*I-0.250*X(a1)*X(a2)-0.250*Y(a1)*Y(a2)-0.250*Z(a1)*Z(a2)",
        ),
        (
            [0.0, 1.0, 1.0j, 0.0],
            "0.250*I-0.250*X(a1)*Y(a2)+0.250*Y(a1)*X(a2)-0.250*Z(a1)*Z(a2)",
        ),
        (
            [0.0, 1.0, -1.0j, 0.0],
            "0.250*I+0.250*X(a1)*Y(a2)-0.250*Y(a1)*X(a2)-0.250*Z(a1)*Z(a2)",
        ),
        (
            [1.0, 0.0, 0.0, 1.0],
            "0.250*I+0.250*X(a1)*X(a2)-0.250*Y(a1)*Y(a2)+0.250*Z(a1)*Z(a2)",
        ),
        (
            [1.0, 0.0, 0.0, -1.0],
            "0.250*I-0.250*X(a1)*X(a2)+0.250*Y(a1)*Y(a2)+0.250*Z(a1)*Z(a2)",
        ),
        (
            [1.0, 0.0, 0.0, 1.0j],
            "0.250*I+0.250*X(a1)*Y(a2)+0.250*Y(a1)*X(a2)+0.250*Z(a1)*Z(a2)",
        ),
        (
            [1.0, 0.0, 0.0, -1.0j],
            "0.250*I-0.250*X(a1)*Y(a2)-0.250*Y(a1)*X(a2)+0.250*Z(a1)*Z(a2)",
        ),
    ),
)
def test_pauli_decomposition_measurement_from_vectors(vector, expected_str):
    a1 = cirq.NamedQubit("a1")
    a2 = cirq.NamedQubit("a2")
    col = np.sqrt(0.5) * np.array(vector).reshape(4, 1)
    measurement = col.dot(col.T.conj())
    decomp = pauli_decomposition(measurement, [a1, a2])
    assert f"{decomp:.3f}" == expected_str

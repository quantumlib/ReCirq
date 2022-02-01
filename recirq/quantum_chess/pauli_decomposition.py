# Copyright 2021 Google
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
"""
Utility functions used for decomposing the given measurement matrix into a PauliSum.
"""
import cirq
import itertools
import functools
import numpy as np
from scipy.linalg import kron
from typing import List


def kron_product(matrices: np.ndarray) -> np.ndarray:
    """Computes the Kronecker product of a given list of matrices."""
    if len(matrices) < 1:
        raise ValueError(
            "kron_product expects a list of matrices to compute Kronecker product."
        )
    return functools.reduce(lambda a, b: kron(a, b), matrices)


def pauli_decomposition(measurement: list, qubits: List[cirq.Qid]) -> cirq.PauliSum:
    """Decompose the given measurement matrix into a PauliSum.

    Args:
    measurement: 2-d matrix with row and column number equals to 2^n (n>0).
       E.g. measurement = [[0.5+0.j,  0. +0.5j], [0. -0.5j,  0.5+0.j]]
    qubits: a list of qubits with size == n.
       E.g. qubits = [a1]
    """
    measurement = np.array(measurement)
    if measurement.ndim != 2:
        raise ValueError("pauli_decomposition expects a 2-d square matrix.")
    d = len(qubits)
    if len(measurement) != 2**d:
        raise ValueError(
            f"pauli_decomposition: Expect that size_of_matrix==pow(2, number_of_qubits). In your case {len(measurement)}!=pow(2, {d})."
        )

    pauli = np.array([cirq.I, cirq.X, cirq.Y, cirq.Z])
    s = np.array([cirq.unitary(op) for op in pauli])

    nonzero = False
    result = 0
    for ind in itertools.product(iter(range(4)), repeat=d):
        ind_array = np.array(ind)
        coeff = (
            0.5**d * np.dot(kron_product(s[ind_array]).T.conj(), measurement).trace()
        )
        if abs(coeff) > 1e-12:
            result = result + cirq.PauliString(
                coeff, {q: op for q, op in zip(qubits, pauli[ind_array])}
            )
            nonzero = True
    if nonzero:
        return result
    return cirq.PauliString(
        0.0, {q: op for q, op in zip(qubits, pauli[np.array([0] * d)])}
    )

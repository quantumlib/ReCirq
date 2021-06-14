# ACopyright 2021 Google
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
Util functions used for measurement.
"""
import cirq
import itertools
import numpy as np
from scipy.linalg import kron
from typing import Iterable

def kron_product(Ms):
    """Computes the Kronecker product of a given list of matrices.
    """
    if len(Ms) < 1:
        raise ValueError("kron_product expects a list of matrices to compute Kronecker product.")
    result = Ms[0]
    for M in Ms[1:]:
        result = kron(result, M)
    return result

def pauli_decomposition(H, qs: Iterable[cirq.Qid]):
    """Decompose the given measurement matrix into PauliStrings. 

    Args:
    H: 2-d matrix with row and column number equals to 2^n (n>0). 
       E.g. H = np.array([[0.5+0.j,  0. +0.5j], [0. -0.5j,  0.5+0.j ]])
ma    qs: a list of qubits with size == n.
       E.g. qs = [a1]
    """
    if H.ndim != 2:
        raise ValueError("pauli_decomposition expects a 2-d matrix.")
    if not all(len(row) == len(H) for row in H):
        raise ValueError("pauli_decomposition expects a matrix with row number equals to column number.")
    d = len(qs)
    if len(H) != 2**d:
        raise ValueError("pauli_decomposition: The list size of qubits is x, and the size of matix is y*y. \
        Expect that y==2**x.")
    
    s0 = np.array([[1.,0.], [0.,1.]])
    s1 = np.array([[0.,1.], [1.,0.]])
    s2 = np.array([[0.,-1.j], [1.j,0.]])
    s3 = np.array([[1.,0.], [0.,-1.]])
    s = np.array([s0, s1, s2, s3])
    pauli = np.array([cirq.I, cirq.X, cirq.Y, cirq.Z])

    result = 0
    for ind in itertools.product(iter(range(4)), repeat=d):
        ind_int = np.array(ind).astype(int)
        coeff = 0.5**d * np.dot(kron_product(s[ind_int]).T.conj(), H).trace()
        if abs(coeff) > 1e-12:
            result = result + cirq.PauliString(coeff, {q: op for q, op in zip(qs, pauli[ind_int])})
    return result


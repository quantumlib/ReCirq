import pytest
import cirq
import numpy as np

from recirq.quantum_chess.measurement_utils import pauli_decomposition

a1 = cirq.NamedQubit('a1')
a2 = cirq.NamedQubit('a2')
a3 = cirq.NamedQubit('a3')
b1 = cirq.NamedQubit('b1')
b2 = cirq.NamedQubit('b2')
b3 = cirq.NamedQubit('b3')
c1 = cirq.NamedQubit('c1')
c2 = cirq.NamedQubit('c2')
c3 = cirq.NamedQubit('c3')

def test_pauli_decomposition_wrong_inputs():
    H_not_2d = np.array([[[0.5+0.j,  0. +0.5j],
                          [0. -0.5j,  0.5+0.j ]]], dtype="object")
    H_not_square = np.array([[0.5+0.j,  0. +0.5j],
                             [0. -0.5j ]], dtype="object")
    H_good = np.array([[0.5+0.j,  0. +0.5j],
                       [0. -0.5j,  0.5+0.j ]], dtype="object")
    with pytest.raises(ValueError):
        pauli_decomposition(H_not_2d, [a1])
        pauli_decomposition(H_not_square, [a1])
        pauli_decomposition(H_good, [a1, a2])
    
def test_pauli_decomposition_1_qubit():
    H = np.random.rand(2,2)
    decomp_H = pauli_decomposition(H, [a1])
    np.allclose(H, decomp_H.matrix())

def test_pauli_decomposition_2_qubit():
    H = np.random.rand(4,4)
    decomp_H = pauli_decomposition(H, [a1, a2])
    np.allclose(H, decomp_H.matrix())

def test_pauli_decomposition_3_qubit():
    H = np.random.rand(8,8)
    decomp_H = pauli_decomposition(H, [a1, a2, a3])
    np.allclose(H, decomp_H.matrix())

def test_pauli_decomposition_4_qubit():
    H = np.random.rand(16,16)
    decomp_H = pauli_decomposition(H, [a1, a2, a3, b1])
    np.allclose(H, decomp_H.matrix())

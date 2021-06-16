import pytest
import cirq
import numpy as np
import re

from recirq.quantum_chess.measurement_utils import pauli_decomposition

def test_pauli_decomposition_wrong_inputs():
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    H_not_2d = np.array([[[0.5+0.j,  0. +0.5j],
                          [0. -0.5j,  0.5+0.j ]]], dtype="object")
    H_not_square = np.array([[0.5+0.j,  0. +0.5j],
                             [0. -0.5j ]], dtype="object")
    H_good = np.array([[0.5+0.j,  0. +0.5j],
                       [0. -0.5j,  0.5+0.j ]], dtype="object")
    with pytest.raises(ValueError, match='pauli_decomposition expects a 2-d square matrix.'):
        pauli_decomposition(H_not_2d, [a1])

    with pytest.raises(ValueError, match='pauli_decomposition expects a 2-d square matrix.'):
        pauli_decomposition(H_not_square, [a1])
        
    with pytest.raises(ValueError, match=re.escape('pauli_decomposition: Expect that size_of_matrix==pow(2, number_of_qubits). In your case 2!=pow(2, 2).')):
        pauli_decomposition(H_good, [a1, a2])
    
def test_pauli_decomposition_1_qubit():
    a1 = cirq.NamedQubit('a1')
    H = np.random.rand(2,2)
    decomp_H = pauli_decomposition(H, [a1])
    assert np.allclose(H, decomp_H.matrix())

def test_pauli_decomposition_2_qubit():
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    H = np.random.rand(4,4)
    decomp_H = pauli_decomposition(H, [a1, a2])
    assert np.allclose(H, decomp_H.matrix())
#    decomp_H = pauli_decomposition(H, [a2, a1])
#    assert np.allclose(H, decomp_H.matrix([a2, a1]))
    
def test_pauli_decomposition_3_qubit():
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    a3 = cirq.NamedQubit('a3')
    H = np.random.rand(8,8)
    decomp_H = pauli_decomposition(H, [a1, a2, a3])
    assert np.allclose(H, decomp_H.matrix())
#    decomp_H = pauli_decomposition(H, [a3, a1, a2])
#    assert np.allclose(H, decomp_H.matrix([a3, a1, a2]))
    
def test_pauli_decomposition_4_qubit():
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    a3 = cirq.NamedQubit('a3')
    b1 = cirq.NamedQubit('b1')
    H = np.random.rand(16,16)
    decomp_H = pauli_decomposition(H, [a1, a2, a3, b1])
    assert np.allclose(H, decomp_H.matrix())
#    decomp_H = pauli_decomposition(H, [a1, b1, a3, a2])
#    assert np.allclose(H, decomp_H.matrix([a1, b1, a3, a2]))

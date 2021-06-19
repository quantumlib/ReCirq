import pytest
import cirq
import numpy as np
import re

from recirq.quantum_chess.pauli_decomposition import pauli_decomposition


def test_pauli_decomposition_wrong_inputs():
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    H_not_2d = [[[0.5+0.j,  0. +0.5j],
                 [0. -0.5j,  0.5+0.j]]]
    H_not_square = [[0.5+0.j,  0. +0.5j],
                    [0. -0.5j]]
    H_good = [[0.5+0.j,  0. +0.5j],
              [0. -0.5j,  0.5+0.j]]
    with pytest.raises(ValueError, match='pauli_decomposition expects a 2-d square matrix.'):
        pauli_decomposition(H_not_2d, [a1])

    with pytest.raises(ValueError, match='pauli_decomposition expects a 2-d square matrix.'):
        pauli_decomposition(H_not_square, [a1])
        
    with pytest.raises(ValueError, match=re.escape('pauli_decomposition: Expect that size_of_matrix==pow(2, number_of_qubits). In your case 2!=pow(2, 2).')):
        pauli_decomposition(H_good, [a1, a2])


@pytest.mark.parametrize(
    'measurement',
    (
        np.random.rand(2,2),
        np.zeros((2, 2)),
    ),
)
def test_pauli_decomposition_1_qubit(measurement):
    a1 = cirq.NamedQubit('a1')
    decomp = pauli_decomposition(measurement, [a1])
    assert np.allclose(measurement, decomp.matrix())


@pytest.mark.parametrize(
    'measurement',
    (
        np.random.rand(4, 4),
        np.zeros((4, 4)),
    ),
)
def test_pauli_decomposition_2_qubit(measurement):
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    for qubits in [[a1, a2], [a2, a1]]:
        decomp = pauli_decomposition(measurement, qubits)
        assert np.allclose(measurement, decomp.matrix())


@pytest.mark.parametrize(
    'measurement',
    (
        np.random.rand(8, 8),
        np.zeros((8, 8)),
    ),
)
def test_pauli_decomposition_3_qubit(measurement):
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    a3 = cirq.NamedQubit('a3')
    for qubits in [[a3, a1, a2]]:
        decomp = pauli_decomposition(measurement, qubits)
        print(decomp)
        decomp2 = decomp.with_qubits(*qubits)
        print(decomp2)
        assert np.allclose(measurement, decomp2.matrix())


@pytest.mark.parametrize(
    'measurement',
    (
        np.random.rand(16, 16),
        np.zeros((16, 16)),
    ),
)
def test_pauli_decomposition_4_qubit(measurement):
    a1 = cirq.NamedQubit('a1')
    a2 = cirq.NamedQubit('a2')
    a3 = cirq.NamedQubit('a3')
    b1 = cirq.NamedQubit('b1')
    for qubits in [[a1, a2, a3, b1], [a1, b1, a3, a2]]:
        decomp = pauli_decomposition(measurement, qubits)
        assert np.allclose(measurement, decomp.matrix())

import cirq
import numpy as np
import pytest
from recirq.third_party import quaff as quaff


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(10)],
)
def test_unitary_round_trip(num_qubits, seed):
    quaff.testing.assert_consistent_unitary_round_trip(
        num_qubits, quaff.BasisChangeGate, seed
    )


@pytest.mark.parametrize(
    "n, seed", [(quaff.RNG.integers(1, 5), quaff.random_seed()) for _ in range(3)]
)
def test_basis_change_pow(n, seed):
    gate = quaff.BasisChangeGate.random(n, seed)
    unitary = cirq.unitary(gate)
    for exponent in range(-3, 4):
        new_gate = gate**exponent
        actual = np.linalg.matrix_power(unitary, exponent)
        expected = cirq.unitary(new_gate)
        assert np.allclose(actual, expected)

    with pytest.raises(ValueError):
        _ = gate**0.3


@pytest.mark.parametrize(
    "n, seed", [(quaff.RNG.integers(1, 10), quaff.random_seed()) for _ in range(10)]
)
def test_clearing_network(n, seed):
    matrix = quaff.random_invertible_matrix(n, seed)
    G = quaff.BasisChangeGate(matrix)
    qubits = cirq.LineQubit.range(n)
    clearing_network, _ = quaff.get_clearing_network(matrix, qubits)
    for op in clearing_network:
        indices = [qubit.x for qubit in op.qubits]
        g = op.gate.expand(n, indices)
        G = g @ G
    assert quaff.is_nw_tri(G.matrix)


def test_eq_bad_type():
    gate = quaff.BasisChangeGate.identity(3)
    with pytest.raises(TypeError):
        _ = gate == 4


def test_str():
    gate = quaff.BasisChangeGate.identity(3)
    assert str(gate) == "BasisChangeGate([[1,0,0],[0,1,0],[0,0,1]])"


def test_from_unitary_bad():
    unitary = np.array([1j])
    with pytest.raises(ValueError):
        quaff.BasisChangeGate.from_unitary(unitary)


def test_copy():
    gate = quaff.BasisChangeGate.identity(2)
    copy = gate.copy()
    assert gate is not copy
    assert gate == copy


def test_from_gate():
    cirq_gate = cirq.CNOT
    basis_change_gate = quaff.BasisChangeGate.from_gate(cirq_gate)
    assert np.allclose(cirq.unitary(basis_change_gate), cirq.unitary(cirq_gate))

    with pytest.raises(NotImplementedError):
        quaff.BasisChangeGate.from_gate(cirq.T)


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(10)]
)
def test_mat_mul(n, seed):
    eye = quaff.BasisChangeGate.identity(n)
    gate = quaff.BasisChangeGate.random(n, seed)
    gate_inv = gate**-1
    assert gate @ gate_inv == eye
    assert gate_inv @ gate == eye


def test_mat_mul_bad():
    with pytest.raises(ValueError):
        _ = quaff.BasisChangeGate.identity(2) @ quaff.BasisChangeGate.identity(3)
    with pytest.raises(TypeError):
        _ = quaff.BasisChangeGate.identity(2) @ 4


@pytest.mark.parametrize(
    "n, seed", [(quaff.RNG.integers(1, 10), quaff.random_seed()) for _ in range(10)]
)
def test_reversal_network(n, seed):
    matrix = quaff.random_invertible_nw_tri_matrix(n, seed)
    G = quaff.BasisChangeGate(matrix)
    qubits = cirq.LineQubit.range(n)
    for op in quaff.get_reversal_network(matrix, qubits):
        indices = [qubit.x for qubit in op.qubits]
        g = op.gate.expand(n, indices)
        G = g @ G
    assert np.array_equal(G.matrix, np.eye(n))


@pytest.mark.parametrize(
    "n, seed", [(quaff.RNG.integers(1, 10), quaff.random_seed()) for _ in range(10)]
)
def test_combined_networks(n, seed):
    matrix = quaff.random_invertible_matrix(n, seed)
    gate = quaff.BasisChangeGate(matrix)
    qubits = cirq.LineQubit.range(n)

    op = gate(*qubits)
    clearing_network, nw_tri_matrix = quaff.get_clearing_network(gate.matrix, qubits)
    reversal_network = list(quaff.get_reversal_network(nw_tri_matrix, qubits))

    unitary = cirq.Circuit([op, clearing_network, reversal_network]).unitary(
        qubit_order=qubits
    )
    assert np.allclose(unitary, np.eye(2**n))

    reversed_networks_circuit = cirq.Circuit(
        [cirq.inverse(reversal_network), cirq.inverse(clearing_network)]
    )
    unitary = reversed_networks_circuit.unitary(qubit_order=qubits)
    assert np.allclose(unitary, cirq.unitary(gate))


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(10)],
)
def test_inverse(num_qubits, seed):
    gate = quaff.BasisChangeGate.random(num_qubits, seed=seed)
    I = quaff.BasisChangeGate.identity(num_qubits)
    assert I == (gate @ gate.inverse())
    assert I == (gate.inverse() @ gate)


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(3, 8) for _ in range(10)],
)
def test_basis_change_gate_decompose(num_qubits, seed):
    gate = quaff.BasisChangeGate.random(num_qubits, seed=seed)
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)

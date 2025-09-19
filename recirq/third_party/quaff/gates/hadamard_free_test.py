import itertools

import cirq
import numpy as np
import pytest
from recirq.third_party import quaff


def assert_hadamard_free_gate_unitary_is_good(n, gate):
    U = cirq.unitary(gate)
    N = 2**n
    assert cirq.is_unitary(U)
    assert U.shape == (N, N)
    for i, j in itertools.product(range(N), repeat=2):
        a = quaff.index_to_bitstring(n, i)
        b = quaff.index_to_bitstring(n, j)
        if np.array_equal(a, ((gate.parity_matrix @ b) + gate.parity_shift) % 2):
            phase = (
                (a @ gate.phase_matrix @ a)
                + (2 * gate.phase_shift @ a)
                + gate.phase_constant
            )
            assert np.isclose(1j**phase, U[i, j])
        else:
            assert np.isclose(U[i, j], 0)


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(25)))
)
def test_hadamard_free_gate_unitary(n, seed):
    gate = quaff.HadamardFreeGate.random(n, seed)
    assert_hadamard_free_gate_unitary_is_good(n, gate)


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(250)))
)
def test_hadamard_free_gate_to_phase_then_parity(n, seed):
    gate = quaff.HadamardFreeGate.random(n, seed)
    phase_gate, parity_gate = gate.to_phase_then_parity()
    assert not phase_gate.changes_parity
    assert not parity_gate.changes_phase
    assert phase_gate.validate()
    assert parity_gate.validate()
    assert gate == parity_gate @ phase_gate


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(25)))
)
def test_hadamard_free_gate_split(n, seed):
    gate = quaff.HadamardFreeGate.random(n, seed)
    parity_gate, phase_gate = gate.split()
    assert not phase_gate.changes_parity
    assert not parity_gate.changes_phase
    assert gate == phase_gate @ parity_gate


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(25)))
)
def test_hadamard_free_gate_inverse(n, seed):
    gate = quaff.HadamardFreeGate.random(n, seed)
    gate_inv = gate.inverse()
    assert gate_inv.validate()
    assert isinstance(gate_inv, quaff.HadamardFreeGate)

    U = cirq.unitary(gate)
    actual = cirq.unitary(gate_inv)
    expected = U.T.conj()
    assert np.allclose(expected, actual)


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(25)))
)
def test_hadamard_free_gate_matmul(n, seed):
    rng = np.random.default_rng(seed)
    G1 = quaff.HadamardFreeGate.random(n, rng)
    G2 = quaff.HadamardFreeGate.random(n, rng)

    G = G2 @ G1
    assert G.validate()
    U1 = cirq.unitary(G1)
    U2 = cirq.unitary(G2)
    expected = U2 @ U1
    actual = cirq.unitary(G)

    assert np.allclose(actual, expected)


@pytest.mark.parametrize(
    "n, seed", ((n, quaff.random_seed()) for n in range(1, 6) for _ in range(5))
)
def test_json(n, seed):
    gate = quaff.HadamardFreeGate.random(n, seed)
    json = cirq.to_json(gate)
    restored_gate = cirq.read_json(json_text=json, resolvers=quaff.DEFAULT_RESOLVERS)
    assert gate == restored_gate

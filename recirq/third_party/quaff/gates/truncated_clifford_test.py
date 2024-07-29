import cirq
import numpy as np
import pytest
from recirq.third_party.quaff import (
    DEFAULT_RESOLVERS,
    HadamardFreeGate,
    TruncatedCliffordGate,
    random_seed,
)


@pytest.mark.parametrize(
    "n, seed", ((n, random_seed()) for n in range(1, 6) for _ in range(5))
)
def test_truncated_clifford_to_fhf_gate(n, seed):
    A = TruncatedCliffordGate.random(n, seed=seed)
    assert A.validate()
    B = A.to_FHF_gate()
    U = cirq.unitary(A)
    V = cirq.unitary(B)

    assert np.allclose(U, V)


@pytest.mark.parametrize("n", range(1, 5))
def test_truncated_clifford_default(n):
    gate = TruncatedCliffordGate(n)
    U = cirq.unitary(gate)
    V = cirq.unitary(HadamardFreeGate(n, parity_matrix=np.flip(np.eye(n), 0)))
    assert np.allclose(U, V)

    gate = TruncatedCliffordGate(n, H=np.ones(n))
    U = cirq.unitary(gate)
    V = cirq.unitary(HadamardFreeGate(n, parity_matrix=np.flip(np.eye(n), 0)))
    H = cirq.kron(*([cirq.unitary(cirq.H)] * n))
    assert np.allclose(U, V @ H)


@pytest.mark.parametrize("n, seed", ((n, random_seed()) for n in range(1, 6)))
def test_truncated_clifford_invalid(n, seed):
    gate = TruncatedCliffordGate.random(n, seed)

    gate_copy = gate.copy()
    gate_copy.P = (0,) * 2 * n
    assert not gate_copy.validate()

    gate_copy = gate.copy()
    gate_copy.CZ = gate_copy.CZ + (0,)
    assert not gate_copy.validate()

    gate_copy = gate.copy()
    gate_copy.H = gate_copy.H + (0,)
    assert not gate_copy.validate()

    gate_copy = gate.copy()
    gate_copy.CX = (0,)
    assert not gate_copy.validate()


def test_truncated_clifford_str():
    gate = TruncatedCliffordGate(2)
    assert str(gate) == "TruncatedCliffordGate(n=2)"

    gate = TruncatedCliffordGate(2, H=(1, 1))
    assert str(gate) == "TruncatedCliffordGate(n=2, H=11)"

    gate = TruncatedCliffordGate(2, H=(1, 1), CZ=(0,))
    assert str(gate) == "TruncatedCliffordGate(n=2, H=11)"

    gate = TruncatedCliffordGate(3, H=(1, 1, 1), CZ=(1, 1, 0))
    assert str(gate) == "TruncatedCliffordGate(n=3, H=111, phase=001|001|110)"

    gate = TruncatedCliffordGate(3, H=(1, 1, 1), CZ=(1, 1, 0), P=(0, 1, 0))
    assert str(gate) == "TruncatedCliffordGate(n=3, H=111, phase=001|011|110)"

    gate = TruncatedCliffordGate(2, CX=((0, 1), (1, 0)))
    assert str(gate) == "TruncatedCliffordGate(n=2, CX=01|10)"


@pytest.mark.parametrize(
    "n, seed", ((n, random_seed()) for n in range(1, 6) for _ in range(5))
)
def test_truncated_clifford_json(n, seed):
    gate = TruncatedCliffordGate.random(n, seed)
    json = cirq.to_json(gate)
    restored_gate = cirq.read_json(json_text=json, resolvers=DEFAULT_RESOLVERS)
    assert gate == restored_gate

import itertools

import cirq
import numpy as np
import pytest
from recirq.third_party import quaff as quaff


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(3)],
)
def test_CZ_layer(num_qubits, seed):
    rng = np.random.default_rng(seed)
    qubits = range(num_qubits)
    num_pairs = num_qubits * (num_qubits - 1) // 2
    pairs = [rng.choice(qubits, 2, replace=False) for _ in range(num_pairs)]
    gate = quaff.CZLayerGate(num_qubits, pairs)
    unitary = gate._unitary_()

    reversed_unitary = quaff.with_qubits_reversed(unitary, num_qubits)
    ne_diagonal = np.diag(reversed_unitary)
    assert np.array_equal(reversed_unitary, np.diag(ne_diagonal))

    for i, x in enumerate(quaff.get_all_bitstrings(num_qubits)):
        parity = sum(x[j] == x[k] == 1 for j, k in pairs)
        assert ne_diagonal[i] == (-1) ** parity


@pytest.mark.parametrize("num_qubits", range(1, 7))
def test_cz_layer_basis_change_stage_gate(num_qubits):
    gate = quaff.CZLayerBasisChangeStageGate(num_qubits)
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(10)],
)
def test_phase_poly(num_qubits, seed):
    rng = np.random.default_rng(seed)
    qubits = range(num_qubits)
    num_pairs = num_qubits * (num_qubits - 1) // 2
    pairs = [rng.choice(qubits, 2, replace=False) for _ in range(num_pairs)]
    gate = quaff.CZLayerGate(num_qubits, pairs)

    phase_poly_in_old_basis = np.zeros((num_qubits,) * 2, dtype=quaff.linalg.DTYPE)
    for i, j in pairs:
        phase_poly_in_old_basis[i, j] += 2

    phase_poly = gate.phase_poly
    for x in quaff.get_all_bitstrings(num_qubits, dtype=int):
        expected = (x @ phase_poly_in_old_basis @ x) % 4
        actual = np.dot(np.diag(phase_poly), x)
        for i, j in itertools.combinations(range(num_qubits), 2):
            u = phase_poly[i, j] + phase_poly[j, i]
            parity = x[i : j + 1].sum() % 2
            actual += u * parity
        actual %= 4
        assert actual == expected


def test_bad_init():
    with pytest.raises(ValueError):
        quaff.CZLayerGate(3, [(0, 0)])
    with pytest.raises(ValueError):
        quaff.CZLayerGate(3, [(0, 5)])
    with pytest.raises(ValueError):
        quaff.CZLayerGate(3, [], False, False)
    with pytest.raises(NotImplementedError):
        quaff.CZLayerGate(3, [], True, False)
    with pytest.raises(NotImplementedError):
        quaff.CZLayerGate(3, [], False, True)


def test_get_index_range_bad():
    with pytest.raises(NotImplementedError):
        quaff.cz_layer.get_index_range(4, 6, 0)


@pytest.mark.parametrize("num_qubits", range(1, 8))
def test_get_index_range(num_qubits):
    qubits = cirq.LineQubit.range(num_qubits)
    for num_steps in range(num_qubits + 2):
        gate = quaff.CZLayerBasisChangeStageGate(num_qubits, num_steps)
        circuit = cirq.Circuit(gate._decompose_(qubits))
        unitary = circuit.unitary(qubit_order=qubits)
        matrix = quaff.BasisChangeGate.from_unitary(unitary).matrix
        assert np.array_equal(gate.basis_change_matrix, matrix)


def test_cz_layer_gate_str():
    gate = quaff.CZLayerGate(4, [(0, 1), (1, 2)])
    assert str(gate) == "CZs(0,1;1,2)"


@pytest.mark.parametrize(
    "num_qubits, seed",
    [(n, quaff.random_seed()) for n in range(1, 6) for _ in range(40)],
)
def test_cz_layer_gate(num_qubits, seed):
    rng = np.random.default_rng(seed)
    num_pairs = num_qubits * (num_qubits - 1) // 2
    indices = range(num_qubits)
    pairs = [rng.choice(indices, 2, replace=False) for _ in range(num_pairs)]
    gate = quaff.CZLayerGate(num_qubits, pairs)

    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


def test_cz_layer_basis_change_stage_str():
    print(str(quaff.CZLayerBasisChangeStageGate(3, 2)))
    assert (
        str(quaff.CZLayerBasisChangeStageGate(3, 2))
        == "CZLayerBasisChangeStageGate(3, 2)"
    )

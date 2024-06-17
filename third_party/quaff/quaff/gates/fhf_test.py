import itertools

import cirq
import numpy as np
import pytest
import quaff


@pytest.mark.parametrize("n, seed", ((n, quaff.random_seed()) for n in range(1, 7)))
def test_FHF_gate_from_sample(n, seed):
    sampler = quaff.CliffordSampler(n)
    rng = np.random.default_rng(seed)
    for _ in range(10):
        sample = sampler.sample(rng)
        print(sample)
        gate = quaff.FHFGate._from_sample(sample)
        assert gate.validate()
        assert gate.is_canonical()


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(25)))
)
def test_FHF_decompose(n, seed):
    gate = quaff.FHFGate.random(n, seed)
    cirq.testing.assert_decompose_is_consistent_with_unitary(gate)


@pytest.mark.parametrize(
    "n, seed", itertools.product(range(1, 5), (quaff.random_seed() for _ in range(25)))
)
def test_FHF_gate_push_earlier(n, seed):
    gate = quaff.FHFGate.random(n, seed)
    assert gate.validate()

    mag = 2 ** (gate.H.sum() / 2)
    U = cirq.unitary(gate) * mag

    new_gate = gate.copy()
    assert gate == new_gate
    assert new_gate.validate()
    new_gate.push_earlier()
    assert new_gate.validate()
    new_U = cirq.unitary(new_gate) * mag

    assert np.array_equal(U, new_U)
    assert new_gate.F2.is_minimal(gate.H)


@pytest.mark.parametrize("n, seed", ((n, quaff.random_seed()) for n in range(1, 8)))
def test_recover_permutation(n, seed):
    rng = np.random.default_rng(seed)
    binom = quaff.binom(n)
    for _ in range(10):
        matrix = quaff.random_tril_and_unit_diag_matrix(n, seed=rng)
        S = rng.permutation(n)
        S_inv = quaff.invert_permutation(S)
        recovered_S = quaff.gates.fhf._recover_permutation(matrix[S_inv, :])
        assert np.array_equal(S, recovered_S)

        triu_bits = rng.integers(0, 2, binom)
        if not triu_bits.any():
            continue
        matrix[quaff.triu_indices(n)] = triu_bits
        assert quaff.gates.fhf._recover_permutation(matrix[S_inv, :]) is None


@pytest.mark.parametrize(
    "n, seed", ((n, quaff.random_seed()) for n in range(1, 6) for _ in range(5))
)
def test_json(n, seed):
    gate = quaff.FHFGate.random(n, seed)
    json = cirq.to_json(gate)
    print(json)
    print(quaff.DEFAULT_RESOLVERS)
    restored_gate = cirq.read_json(json_text=json, resolvers=quaff.DEFAULT_RESOLVERS)
    assert gate == restored_gate

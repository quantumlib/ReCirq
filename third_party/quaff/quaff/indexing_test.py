import itertools

import numpy as np
import pytest
import quaff


@pytest.mark.parametrize("n, big_endian", itertools.product(range(5), (0, 1)))
def test_get_all_bitstrings(n, big_endian):
    bitstrings = quaff.get_all_bitstrings(n, big_endian)
    N = 2**n
    assert bitstrings.shape == (N, n)
    masks = np.arange(n)
    if big_endian:
        masks = masks[::-1]
    actual = bitstrings @ (2**masks)[:, np.newaxis]
    expected = np.arange(N)[:, np.newaxis]
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("n", range(1, 10))
def test_singular_indexing(n):
    bitstrings = quaff.get_all_bitstrings(n)
    N = 2**n
    for i in range(N):
        assert i == quaff.bitstring_to_index(quaff.index_to_bitstring(n, i))
    for x in bitstrings:
        assert np.array_equal(
            x, quaff.index_to_bitstring(n, quaff.bitstring_to_index(x))
        )

    unit_vector_indices = quaff.get_unit_vector_indices(n)
    assert unit_vector_indices.shape == (n,)
    for i, j in enumerate(unit_vector_indices):
        assert quaff.get_unit_vector_index(n, i) == j


@pytest.mark.parametrize("n, seed", [(n, quaff.random_seed()) for n in (2, 5, 10)])
def test_plural_indexing(n, seed):
    rng = np.random.default_rng(seed)
    bitstrings = [b for b in itertools.product((0, 1), repeat=n) if rng.random() < 0.3]
    if not bitstrings:
        return  # pragma: no cover
    indices = quaff.bitstrings_to_indices(bitstrings)
    for i, b in enumerate(bitstrings):
        assert indices[i] == quaff.bitstring_to_index(b)


@pytest.mark.parametrize("n, seed", [(n, quaff.random_seed()) for n in range(1, 10)])
def test_offdiag_indices(n, seed):
    matrix = np.random.default_rng(seed).random((n, n))
    diag_indices = np.diag_indices(n)
    offdiag_indices = quaff.offdiag_indices(n)
    new_matrix = np.zeros_like(matrix)
    new_matrix[diag_indices] += matrix[diag_indices]
    new_matrix[offdiag_indices] += matrix[offdiag_indices]
    assert np.allclose(matrix, new_matrix)


def test_log2():
    for n in range(1, 10):
        assert n == quaff.log2(2**n)

    for N in (1.3, 3, 5, 5.2):
        with pytest.raises(ValueError):
            quaff.log2(N)


@pytest.mark.parametrize(
    "n, seed", ((quaff.RNG.integers(1, 10), quaff.random_seed()) for _ in range(10))
)
def test_tri_indices(n, seed):
    rng = np.random.default_rng(seed)
    binom = quaff.binom(n)
    bits = rng.integers(0, 5, binom)
    matrix = np.zeros((n, n), dtype=quaff.DTYPE)
    tril_indices = quaff.tril_indices(n)
    assert len(tril_indices) == 2
    assert tril_indices[0].shape == tril_indices[1].shape == (binom,)
    triu_indices = quaff.triu_indices(n)
    assert len(triu_indices) == 2
    assert triu_indices[0].shape == triu_indices[1].shape == (binom,)
    matrix[tril_indices] = bits
    matrix[triu_indices] = bits
    assert np.array_equal(matrix, matrix.T)

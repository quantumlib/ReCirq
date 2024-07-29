import numpy as np
import pytest
from recirq.third_party import quaff as quaff


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in range(1, 10) for _ in range(10)]
)
def test_random_invertible_nw_tri_matrix(n, seed):
    matrix = quaff.random_invertible_nw_tri_matrix(n, seed)
    assert quaff.is_nw_tri(matrix)
    _, pivots = quaff.row_reduce(matrix)
    assert len(pivots) == n


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in (2, 5, 10) for _ in range(10)]
)
def test_random_nw_tri_matrix(n, seed):
    matrix = quaff.random_nw_tri_matrix(n, seed)
    assert quaff.is_nw_tri(matrix)


@pytest.mark.parametrize("n, seed", [(n, quaff.random_seed()) for n in (2, 5, 10)])
def test_random_symmetric_matrix(n, seed):
    matrix = quaff.random_symmetric_matrix(n, seed=seed)
    assert np.allclose(matrix, matrix.T)


@pytest.mark.parametrize(
    "n, seed", [(n, quaff.random_seed()) for n in range(1, 5) for _ in range(3)]
)
def test_random_permutation(n, seed):
    perm = quaff.random_permutation(n, seed)
    assert np.array_equal(np.sort(perm), np.arange(n))

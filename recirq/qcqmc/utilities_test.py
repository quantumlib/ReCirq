import numpy as np

from recirq.qcqmc.utilities import (
    iterate_permutation_matrices,
    iterate_virtual_permutation_matrices,
)


def test_iterate_permutation_matrices():
    dim_1_matrices = list(iterate_permutation_matrices(1))
    assert len(dim_1_matrices) == 1

    dim_2_matrices = list(iterate_permutation_matrices(2))
    assert len(dim_2_matrices) == 2

    dim_3_matrices = list(iterate_permutation_matrices(3))
    assert len(dim_3_matrices) == 6

    for mat in dim_3_matrices:
        np.testing.assert_array_almost_equal(
            np.diag(np.ones((3,))), mat @ np.transpose(mat)
        )


def test_iterate_virtual_permutation_matrices():
    matrices = list(iterate_virtual_permutation_matrices(4, 4))

    print(matrices)

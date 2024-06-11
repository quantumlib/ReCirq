import numpy as np
import pytest
import scipy

from recirq.qcqmc.bitstrings import get_bitstrings_a_b


def test_get_bitstrings_a_b():
    with pytest.raises(NotImplementedError):
        list(get_bitstrings_a_b(n_orb=4, n_elec=3))

    bitstrings = np.array(list(get_bitstrings_a_b(n_orb=4, n_elec=4)))

    assert bitstrings.shape[0] == scipy.special.binom(4, 2) ** 2
    assert bitstrings.shape[1] == 2 * 4  # n_qubits columns = 2 * n_orb.
    hamming_weight_left = np.sum(bitstrings[:, 0:4], axis=1)
    hamming_weight_right = np.sum(bitstrings[:, 4:8], axis=1)

    assert np.all(hamming_weight_left == 2)
    assert np.all(hamming_weight_right == 2)

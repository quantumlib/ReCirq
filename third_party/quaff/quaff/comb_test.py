import collections
import itertools
import math

import numpy as np
import pytest
import quaff


def test_get_num_perms_with_inversion_number():
    max_n = 8
    num_perms_with_inversion_number = quaff.get_num_perms_with_inversion_number(max_n)
    for n in range(1, max_n + 1):
        binom = quaff.binom(n)
        total = math.factorial(n)
        counter = collections.Counter(
            quaff.get_inversion_number(perm)
            for perm in itertools.permutations(range(n))
        )
        assert sum(counter.values()) == total
        assert (
            sum(num_perms_with_inversion_number[n, i] for i in range(binom + 1))
            == total
        )
        for i in range(binom + 1):
            assert counter[i] == num_perms_with_inversion_number[n, i]


@pytest.mark.parametrize("n", range(1, 9))
def test_get_mahonian_numbers(n):
    mahonian_numbers = quaff.comb.get_mahonian_numbers(n)

    binom = quaff.binom(n)
    total = math.factorial(n)
    counter = collections.Counter(
        quaff.get_inversion_number(perm) for perm in itertools.permutations(range(n))
    )
    assert sum(counter.values()) == total
    assert mahonian_numbers.sum() == total
    assert np.array_equal([counter[i] for i in range(binom + 1)], mahonian_numbers)

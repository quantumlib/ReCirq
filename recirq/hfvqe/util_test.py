import numpy as np
import scipy as sp
from openfermioncirq.experiments.hfvqe.util import (generate_permutations,
                        swap_forward,
                        generate_fswap_pairs,
                        generate_fswap_unitaries)


def test_swap_forward():
    list_to_swap = list(range(6))
    test_swapped_list = swap_forward(list_to_swap, starting_index=0)
    assert test_swapped_list == [1, 0, 3, 2, 5, 4]

    test_swapped_list = swap_forward(list_to_swap, starting_index=1)
    assert test_swapped_list == [0, 2, 1, 4, 3, 5]


def test_generate_fswap_pairs():
    swap_set = generate_fswap_pairs(2, 6)
    assert swap_set[0] == [(0, 1), (2, 3), (4, 5)]
    assert swap_set[1] == [(1, 2), (3, 4)]

    swap_set = generate_fswap_pairs(1, 4)
    assert swap_set[0] == [(0, 1), (2, 3)]


def test_gen_fswap_unitaries():
    fswapu = generate_fswap_unitaries([((0, 1), (2, 3))], 4)
    true_generator = np.zeros((4, 4), dtype=np.complex128)
    true_generator[0, 0], true_generator[1, 1] = -1, -1
    true_generator[0, 1], true_generator[1, 0] = 1, 1
    true_generator[2, 2], true_generator[3, 3] = -1, -1
    true_generator[2, 3], true_generator[3, 2] = 1, 1
    true_u = sp.linalg.expm(-1j * np.pi * true_generator / 2)
    assert np.allclose(true_u, fswapu[0])


def test_permutation_generator():
    perms = generate_permutations(3)
    assert len(perms) == 2  # N//2+1 circuits
    assert perms[0] == [0, 1, 2]
    assert perms[1] == [1, 2, 0]

    perms = generate_permutations(4)
    assert len(perms) == 2  # N/2 circuits
    assert perms[0] == [0, 1, 2, 3]
    assert perms[1] == [1, 3, 0, 2]

    perms = generate_permutations(6)
    assert len(perms) == 3  # N/2 circuits
    assert perms[0] == [0, 1, 2, 3, 4, 5]
    assert perms[1] == [1, 3, 0, 5, 2, 4]
    assert perms[2] == [3, 5, 1, 4, 0, 2]

    perms = generate_permutations(4, no_truncation=True)
    assert len(perms) == 5
    assert perms[1] == [1, 0, 3, 2]
    assert perms[3] == [3, 1, 2, 0]

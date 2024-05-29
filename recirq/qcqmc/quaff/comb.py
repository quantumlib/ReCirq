import collections
import itertools
import math
from typing import Dict, Tuple

import numpy as np
import scipy.special


def binom(n: int) -> int:
    return scipy.special.comb(n, 2, exact=True)


def get_inversion_number(perm: np.ndarray) -> int:
    return sum(perm[i] > perm[j] for i, j in itertools.combinations(range(len(perm)), 2))


def get_num_perms_with_inversion_number(max_n: int) -> Dict[Tuple[int, int], int]:
    # TODO: replace with mahonian numbers
    num_perms = collections.defaultdict(int)
    num_perms[1, 0] = 1
    num_perms[2, 0] = 1
    num_perms[2, 1] = 1
    for n in range(2, max_n + 1):
        for k in range(binom(n) + 1):
            num_perms[n, k] = num_perms[n, k - 1] + num_perms[n - 1, k] - num_perms[n - 1, k - n]
    return num_perms


def get_mahonian_numbers(n: int) -> np.ndarray:
    N = binom(n) + 1
    mahonian_numbers = np.eye(N, 1, dtype=int).flatten()
    for j in range(n):
        mahonian_numbers = sum(np.eye(N, N, -k, dtype=int) for k in range(j + 1)) @ mahonian_numbers
    return mahonian_numbers


def num_invertible_matrices(n: int) -> int:
    return math.prod(2**i - 1 for i in range(1, n + 1)) * (2 ** binom(n))


def num_cliffords(n: int) -> int:
    return math.prod(4**i - 1 for i in range(1, n + 1)) * (2 ** (n * (n + 2)))

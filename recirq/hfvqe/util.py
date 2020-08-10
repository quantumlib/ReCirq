from typing import Iterable, Optional, List, Tuple
import copy
import numpy as np
from scipy.linalg import expm


def generate_permutations(n_orbitals: int,
                          no_truncation: Optional[bool] = False):
    qubit_orderings = [list(range(n_orbitals))]
    if n_orbitals % 2 == 0:
        perm_order = n_orbitals // 2
    else:
        perm_order = n_orbitals // 2 + 1
    for _ in range(perm_order):
        qubit_orderings.append(swap_forward(qubit_orderings[-1],
                                            starting_index=0))
        qubit_orderings.append(swap_forward(qubit_orderings[-1],
                                            starting_index=1))
    if no_truncation:
        return qubit_orderings
    else:
        return qubit_orderings[::2][:-1]


def swap_forward(iterable_item: Iterable,
                 starting_index: Optional[int] = 0):
    new_sequence = copy.deepcopy(iterable_item)
    for i in range(starting_index, len(iterable_item) - 1, 2):
        new_sequence[i + 1], new_sequence[i] = \
            new_sequence[i], new_sequence[i + 1]
    return new_sequence


def generate_fswap_pairs(depth: int, dimension: int):
    swap_list = []
    for i in range(0, depth):
        if i % 2 == 0:
            swap_list.append([(i, i + 1) for i in range(0, dimension - 1, 2)])
        else:
            swap_list.append([(i, i + 1) for i in range(1, dimension - 1, 2)])
    return swap_list


def generate_fswap_unitaries(swap_pairs: List[List[Tuple]], dimension: int):
    swap_unitaries = []
    for swap_tuples in swap_pairs:
        generator = np.zeros((dimension, dimension), dtype=np.complex128)
        for i, j in swap_tuples:
            generator[i, i] = -1
            generator[j, j] = -1
            generator[i, j] = 1
            generator[j, i] = 1
        swap_unitaries.append(expm(-1j * np.pi * generator / 2))
    return swap_unitaries
